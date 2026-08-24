//===---------- DeviceOffload.cpp - Device Offloading------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements offloading to CUDA devices.
//
//===----------------------------------------------------------------------===//

#include "DeviceOffload.h"
#include "IncrementalAction.h"

#include "clang/Basic/TargetOptions.h"
#include "clang/CodeGen/ModuleBuilder.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Interpreter/PartialTranslationUnit.h"

#include "llvm/ADT/StringSet.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Module.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/Linker/Linker.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/FileUtilities.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Program.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/TargetParser/Host.h"
#include "llvm/Transforms/IPO/Internalize.h"

namespace clang {

static llvm::Expected<llvm::TargetMachine *>
getOrCreateTargetMachine(std::unique_ptr<llvm::TargetMachine> &Cache,
                         llvm::Module &M, llvm::StringRef CPU) {
  if (!Cache) {
    std::string Error;
    const llvm::Target *Target =
        llvm::TargetRegistry::lookupTarget(M.getTargetTriple(), Error);
    if (!Target)
      return llvm::make_error<llvm::StringError>(std::move(Error),
                                                 std::error_code());
    llvm::TargetOptions TO = llvm::TargetOptions();
    Cache.reset(Target->createTargetMachine(M.getTargetTriple(), CPU, "", TO,
                                            llvm::Reloc::Model::PIC_));
  }
  M.setDataLayout(Cache->createDataLayout());
  return Cache.get();
}

IncrementalHIPDeviceParser::IncrementalHIPDeviceParser(
    CompilerInstance &DeviceInstance, CompilerInstance &HostInstance,
    IncrementalAction *DeviceAct,
    llvm::IntrusiveRefCntPtr<llvm::vfs::InMemoryFileSystem> FS,
    llvm::Error &Err, std::list<PartialTranslationUnit> &PTUs)
    : IncrementalParser(DeviceInstance, DeviceAct, Err, PTUs), VFS(FS),
      CodeGenOpts(HostInstance.getCodeGenOpts()),
      DeviceCodeGenOpts(DeviceInstance.getCodeGenOpts()),
      TargetOpts(DeviceInstance.getTargetOpts()) {
  if (Err)
    return;
  StringRef Arch = TargetOpts.CPU;
  if (!Arch.starts_with("gfx")) {
    Err = llvm::joinErrors(std::move(Err), llvm::make_error<llvm::StringError>(
                                               "Invalid HIP architecture",
                                               llvm::inconvertibleErrorCode()));
    return;
  }
}

llvm::Error IncrementalHIPDeviceParser::optimize() {
  auto &PTU = PTUs.back();

  llvm::Expected<llvm::TargetMachine *> TMOrErr =
      getOrCreateTargetMachine(TM, *PTU.TheModule, TargetOpts.CPU);
  if (!TMOrErr)
    return TMOrErr.takeError();

  llvm::LoopAnalysisManager LAM;
  llvm::FunctionAnalysisManager FAM;
  llvm::CGSCCAnalysisManager CGAM;
  llvm::ModuleAnalysisManager MAM;

  llvm::PassBuilder PB(*TMOrErr);
  PB.registerModuleAnalyses(MAM);
  PB.registerCGSCCAnalyses(CGAM);
  PB.registerFunctionAnalyses(FAM);
  PB.registerLoopAnalyses(LAM);
  PB.crossRegisterProxies(LAM, FAM, CGAM, MAM);

  llvm::OptimizationLevel OptLevel;
  switch (DeviceCodeGenOpts.OptimizationLevel) {
  case 0:
    OptLevel = llvm::OptimizationLevel::O0;
    break;
  case 1:
    OptLevel = llvm::OptimizationLevel::O1;
    break;
  case 2:
    OptLevel = llvm::OptimizationLevel::O2;
    break;
  default:
    OptLevel = llvm::OptimizationLevel::O3;
    break;
  }

  llvm::ModulePassManager MPM =
      OptLevel == llvm::OptimizationLevel::O0
          ? PB.buildO0DefaultPipeline(OptLevel)
          : PB.buildPerModuleDefaultPipeline(OptLevel);
  MPM.run(*PTU.TheModule, MAM);
  return llvm::Error::success();
}

llvm::Expected<llvm::StringRef> IncrementalHIPDeviceParser::GenerateHSACO() {
  auto &PTU = PTUs.back();

  llvm::Expected<llvm::TargetMachine *> TMOrErr =
      getOrCreateTargetMachine(TM, *PTU.TheModule, TargetOpts.CPU);
  if (!TMOrErr)
    return TMOrErr.takeError();
  llvm::TargetMachine *TargetMachine = *TMOrErr;

  llvm::SmallVector<char, 0> Object;
  llvm::raw_svector_ostream ObjOS(Object);

  llvm::legacy::PassManager PM;
  if (TargetMachine->addPassesToEmitFile(PM, ObjOS, nullptr,
                                         llvm::CodeGenFileType::ObjectFile))
    return llvm::make_error<llvm::StringError>(
        "AMDGPU backend cannot produce an object file.",
        llvm::inconvertibleErrorCode());

  if (!PM.run(*PTU.TheModule))
    return llvm::make_error<llvm::StringError>(
        "Failed to emit the object file.", llvm::inconvertibleErrorCode());

  // Link the object into a shared .hsaco code object with ld.lld.
  std::string Exe = llvm::sys::fs::getMainExecutable(nullptr, nullptr);
  llvm::StringRef ExeDir = llvm::sys::path::parent_path(Exe);
  llvm::ErrorOr<std::string> LLDPath =
      llvm::sys::findProgramByName("ld.lld", {ExeDir});
  if (!LLDPath)
    LLDPath = llvm::sys::findProgramByName("ld.lld");
  if (!LLDPath)
    return llvm::make_error<llvm::StringError>(
        "Could not find ld.lld next to the executable or on PATH.",
        llvm::inconvertibleErrorCode());

  int ObjFD = -1;
  llvm::SmallString<128> ObjFile;
  if (llvm::sys::fs::createTemporaryFile("kernel", "o", ObjFD, ObjFile))
    return llvm::make_error<llvm::StringError>(
        "Failed to create a temporary object file.",
        llvm::inconvertibleErrorCode());
  llvm::FileRemover ObjRemover(ObjFile);
  {
    llvm::raw_fd_ostream OS(ObjFD, /*shouldClose=*/true);
    OS << llvm::StringRef(Object.data(), Object.size());
  }

  llvm::SmallString<128> HsacoFile;
  if (llvm::sys::fs::createTemporaryFile("kernel", "hsaco", HsacoFile))
    return llvm::make_error<llvm::StringError>(
        "Failed to create a temporary code object file.",
        llvm::inconvertibleErrorCode());
  llvm::FileRemover HsacoRemover(HsacoFile);

  llvm::StringRef Args[] = {"ld.lld", "-shared", ObjFile, "-o", HsacoFile};
  if (llvm::sys::ExecuteAndWait(*LLDPath, Args) != 0)
    return llvm::make_error<llvm::StringError>("ld.lld invocation failed.",
                                               llvm::inconvertibleErrorCode());

  auto HsacoBuf = llvm::MemoryBuffer::getFile(HsacoFile, /*IsText=*/false);
  if (!HsacoBuf)
    return llvm::make_error<llvm::StringError>(
        "Failed to read the code object.", llvm::inconvertibleErrorCode());

  llvm::StringRef Buffer = (*HsacoBuf)->getBuffer();
  HSACOContent.assign(Buffer.begin(), Buffer.end());
  return llvm::StringRef(HSACOContent.data(), HSACOContent.size());
}

llvm::Error IncrementalHIPDeviceParser::GenerateOffloadBundle() {
  // The host embeds this blob as __hip_fatbin; __hipRegisterFatBinary parses
  // it as a clang-offload-bundle:
  //   char     Magic["__CLANG_OFFLOAD_BUNDLE__"]  (no NUL terminator)
  //   uint64_t NumberOfEntries
  //   for each entry: uint64_t Offset, Size, TripleSize; char Triple[]
  //   the code objects follow; HIP requires each to be page-aligned (4096).
  static constexpr llvm::StringRef Magic = "__CLANG_OFFLOAD_BUNDLE__";
  static constexpr uint64_t CodeObjectAlign = 4096;

  const PartialTranslationUnit &PTU = PTUs.back();
  // Triples use the normalized 4-field form ending in a dash; the device entry
  // additionally appends the offload arch, e.g.
  // "hip-amdgcn-amd-amdhsa--gfx90a".
  std::string HostTriple = "host-" + llvm::sys::getProcessTriple() + "-";
  std::string DeviceTriple =
      "hip-" + PTU.TheModule->getTargetTriple().str() + "--" + TargetOpts.CPU;

  const uint64_t NumEntries = 2;
  const uint64_t HeaderSize = Magic.size() + sizeof(uint64_t) +
                              NumEntries * (3 * sizeof(uint64_t)) +
                              HostTriple.size() + DeviceTriple.size();
  const uint64_t CodeObjectOffset = llvm::alignTo(HeaderSize, CodeObjectAlign);

  llvm::SmallVector<char, 4096> Bundle;
  llvm::raw_svector_ostream OS(Bundle);
  auto WriteU64 = [&OS](uint64_t V) {
    OS.write(reinterpret_cast<const char *>(&V), sizeof(V));
  };

  OS << Magic;
  WriteU64(NumEntries);

  // Host entry: empty content. Not required (the runtime matches by triple and
  // skips host entries); emitted only to mirror the canonical bundler layout,
  // and free since page alignment keeps the bundle the same size regardless.
  WriteU64(CodeObjectOffset);
  WriteU64(/*Size=*/0);
  WriteU64(HostTriple.size());
  OS << HostTriple;

  // Device entry: the page-aligned .hsaco content.
  WriteU64(CodeObjectOffset);
  WriteU64(HSACOContent.size());
  WriteU64(DeviceTriple.size());
  OS << DeviceTriple;

  // Zero-pad the header up to the aligned code-object offset.
  OS << std::string(CodeObjectOffset - HeaderSize, '\0');
  OS << llvm::StringRef(HSACOContent.data(), HSACOContent.size());

  std::string BundleFileName = "/" + PTU.TheModule->getName().str() + ".hipfb";
  VFS->addFile(BundleFileName, 0,
               llvm::MemoryBuffer::getMemBufferCopy(
                   llvm::StringRef(Bundle.data(), Bundle.size())));

  CodeGenOpts.OffloadBinaryToEmbedFile = std::move(BundleFileName);
  return llvm::Error::success();
}

IncrementalHIPDeviceParser::~IncrementalHIPDeviceParser() {}

IncrementalCUDADeviceParser::IncrementalCUDADeviceParser(
    CompilerInstance &DeviceInstance, CompilerInstance &HostInstance,
    IncrementalAction *DeviceAct,
    llvm::IntrusiveRefCntPtr<llvm::vfs::InMemoryFileSystem> FS,
    llvm::Error &Err, std::list<PartialTranslationUnit> &PTUs)
    : IncrementalParser(DeviceInstance, DeviceAct, Err, PTUs), VFS(FS),
      CodeGenOpts(HostInstance.getCodeGenOpts()),
      TargetOpts(DeviceInstance.getTargetOpts()) {
  if (Err)
    return;
  StringRef Arch = TargetOpts.CPU;
  if (!Arch.starts_with("sm_") || Arch.substr(3).getAsInteger(10, SMVersion)) {
    Err = llvm::joinErrors(std::move(Err), llvm::make_error<llvm::StringError>(
                                               "Invalid CUDA architecture",
                                               llvm::inconvertibleErrorCode()));
    return;
  }
}

llvm::Expected<llvm::StringRef> IncrementalCUDADeviceParser::GeneratePTX() {
  auto &PTU = PTUs.back();

  llvm::Expected<llvm::TargetMachine *> TMOrErr =
      getOrCreateTargetMachine(TM, *PTU.TheModule, TargetOpts.CPU);
  if (!TMOrErr)
    return TMOrErr.takeError();
  llvm::TargetMachine *TargetMachine = *TMOrErr;

  PTXCode.clear();
  llvm::raw_svector_ostream dest(PTXCode);

  llvm::legacy::PassManager PM;
  if (TargetMachine->addPassesToEmitFile(PM, dest, nullptr,
                                         llvm::CodeGenFileType::AssemblyFile)) {
    return llvm::make_error<llvm::StringError>(
        "NVPTX backend cannot produce PTX code.",
        llvm::inconvertibleErrorCode());
  }

  if (!PM.run(*PTU.TheModule))
    return llvm::make_error<llvm::StringError>("Failed to emit PTX code.",
                                               llvm::inconvertibleErrorCode());

  PTXCode += '\0';
  while (PTXCode.size() % 8)
    PTXCode += '\0';
  return PTXCode.str();
}

llvm::Error IncrementalCUDADeviceParser::GenerateFatbinary() {
  enum FatBinFlags {
    AddressSize64 = 0x01,
    HasDebugInfo = 0x02,
    ProducerCuda = 0x04,
    HostLinux = 0x10,
    HostMac = 0x20,
    HostWindows = 0x40
  };

  struct FatBinInnerHeader {
    uint16_t Kind;             // 0x00
    uint16_t unknown02;        // 0x02
    uint32_t HeaderSize;       // 0x04
    uint32_t DataSize;         // 0x08
    uint32_t unknown0c;        // 0x0c
    uint32_t CompressedSize;   // 0x10
    uint32_t SubHeaderSize;    // 0x14
    uint16_t VersionMinor;     // 0x18
    uint16_t VersionMajor;     // 0x1a
    uint32_t CudaArch;         // 0x1c
    uint32_t unknown20;        // 0x20
    uint32_t unknown24;        // 0x24
    uint32_t Flags;            // 0x28
    uint32_t unknown2c;        // 0x2c
    uint32_t unknown30;        // 0x30
    uint32_t unknown34;        // 0x34
    uint32_t UncompressedSize; // 0x38
    uint32_t unknown3c;        // 0x3c
    uint32_t unknown40;        // 0x40
    uint32_t unknown44;        // 0x44
    FatBinInnerHeader(uint32_t DataSize, uint32_t CudaArch, uint32_t Flags)
        : Kind(1 /*PTX*/), unknown02(0x0101), HeaderSize(sizeof(*this)),
          DataSize(DataSize), unknown0c(0), CompressedSize(0),
          SubHeaderSize(HeaderSize - 8), VersionMinor(2), VersionMajor(4),
          CudaArch(CudaArch), unknown20(0), unknown24(0), Flags(Flags),
          unknown2c(0), unknown30(0), unknown34(0), UncompressedSize(0),
          unknown3c(0), unknown40(0), unknown44(0) {}
  };

  struct FatBinHeader {
    uint32_t Magic;      // 0x00
    uint16_t Version;    // 0x04
    uint16_t HeaderSize; // 0x06
    uint32_t DataSize;   // 0x08
    uint32_t unknown0c;  // 0x0c
  public:
    FatBinHeader(uint32_t DataSize)
        : Magic(0xba55ed50), Version(1), HeaderSize(sizeof(*this)),
          DataSize(DataSize), unknown0c(0) {}
  };

  FatBinHeader OuterHeader(sizeof(FatBinInnerHeader) + PTXCode.size());
  FatbinContent.append((char *)&OuterHeader,
                       ((char *)&OuterHeader) + OuterHeader.HeaderSize);

  FatBinInnerHeader InnerHeader(PTXCode.size(), SMVersion,
                                FatBinFlags::AddressSize64 |
                                    FatBinFlags::HostLinux);
  FatbinContent.append((char *)&InnerHeader,
                       ((char *)&InnerHeader) + InnerHeader.HeaderSize);

  FatbinContent.append(PTXCode.begin(), PTXCode.end());

  const PartialTranslationUnit &PTU = PTUs.back();

  std::string FatbinFileName = "/" + PTU.TheModule->getName().str() + ".fatbin";

  VFS->addFile(FatbinFileName, 0,
               llvm::MemoryBuffer::getMemBuffer(
                   llvm::StringRef(FatbinContent.data(), FatbinContent.size()),
                   "", false));

  CodeGenOpts.OffloadBinaryToEmbedFile = std::move(FatbinFileName);

  FatbinContent.clear();

  return llvm::Error::success();
}

IncrementalCUDADeviceParser::~IncrementalCUDADeviceParser() {}

} // namespace clang
