//===---------- DeviceOffload.cpp - Device Offloading------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements offloading to HIP and CUDA devices.
//
//===----------------------------------------------------------------------===//

#include "DeviceOffload.h"
#include "IncrementalAction.h"

#include "clang/Basic/TargetOptions.h"
#include "clang/CodeGen/BackendUtil.h"
#include "clang/CodeGen/CodeGenAction.h"
#include "clang/CodeGen/ModuleBuilder.h"
#include "clang/Driver/OffloadBundler.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/FrontendAction.h"
#include "clang/Interpreter/PartialTranslationUnit.h"

#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Module.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/FileUtilities.h"
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
    : IncrementalParser(DeviceInstance, DeviceAct, Err, PTUs),
      DeviceCI(DeviceInstance), VFS(FS),
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

llvm::Expected<TranslationUnitDecl *>
IncrementalHIPDeviceParser::Parse(llvm::StringRef Input) {
  if (FrontendAction *WrappedAct = Act->getWrapped())
    if (WrappedAct->hasIRSupport())
      static_cast<CodeGenAction *>(WrappedAct)->reloadLinkModules(DeviceCI);

  return IncrementalParser::Parse(Input);
}

llvm::Expected<llvm::StringRef> IncrementalHIPDeviceParser::GenerateHSACO() {
  auto &PTU = PTUs.back();

  llvm::SmallVector<char, 0> Object;
  auto ObjOS = std::make_unique<llvm::raw_svector_ostream>(Object);
  clang::emitBackendOutput(
      DeviceCI, DeviceCI.getCodeGenOpts(),
      DeviceCI.getTarget().getDataLayoutString(), PTU.TheModule.get(),
      Backend_EmitObj, DeviceCI.getVirtualFileSystemPtr(), std::move(ObjOS));

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

  llvm::StringRef Args[] = {"ld.lld", "-shared", "--no-undefined",
                            ObjFile,  "-o",      HsacoFile};
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
  static constexpr unsigned CodeObjectAlign = 4096;

  const PartialTranslationUnit &PTU = PTUs.back();

  llvm::SmallString<128> HostFile;
  if (llvm::sys::fs::createTemporaryFile("hip-host", "", HostFile))
    return llvm::make_error<llvm::StringError>(
        "Failed to create a temporary host bundle input.",
        llvm::inconvertibleErrorCode());
  llvm::FileRemover HostRemover(HostFile);

  llvm::SmallString<128> DeviceFile;
  int DeviceFD = -1;
  if (llvm::sys::fs::createTemporaryFile("hip-device", "hsaco", DeviceFD,
                                         DeviceFile))
    return llvm::make_error<llvm::StringError>(
        "Failed to create a temporary code object file.",
        llvm::inconvertibleErrorCode());
  llvm::FileRemover DeviceRemover(DeviceFile);
  {
    llvm::raw_fd_ostream OS(DeviceFD, /*shouldClose=*/true);
    OS << llvm::StringRef(HSACOContent.data(), HSACOContent.size());
  }

  llvm::SmallString<128> BundleFile;
  if (llvm::sys::fs::createTemporaryFile("hip-bundle", "hipfb", BundleFile))
    return llvm::make_error<llvm::StringError>(
        "Failed to create a temporary offload bundle file.",
        llvm::inconvertibleErrorCode());
  llvm::FileRemover BundleRemover(BundleFile);

  // Triples use the normalized 4-field form ending in a dash; the device entry
  // additionally appends the offload arch, e.g.
  // "hip-amdgcn-amd-amdhsa--gfx90a".
  std::string HostTriple = "host-" + llvm::sys::getProcessTriple() + "-";
  std::string DeviceTriple =
      "hip-" + PTU.TheModule->getTargetTriple().str() + "--" + TargetOpts.CPU;

  OffloadBundlerConfig Config;
  Config.FilesType = "o";
  Config.BundleAlignment = CodeObjectAlign;
  Config.HostInputIndex = 0;
  Config.TargetNames = {HostTriple, DeviceTriple};
  Config.InputFileNames = {std::string(HostFile), std::string(DeviceFile)};
  Config.OutputFileNames = {std::string(BundleFile)};

  if (llvm::Error Err = OffloadBundler(Config).BundleFiles())
    return Err;

  auto BundleBuf = llvm::MemoryBuffer::getFile(BundleFile, /*IsText=*/false);
  if (!BundleBuf)
    return llvm::make_error<llvm::StringError>(
        "Failed to read the offload bundle.", llvm::inconvertibleErrorCode());

  std::string BundleFileName = "/" + PTU.TheModule->getName().str() + ".hipfb";
  VFS->addFile(BundleFileName, 0,
               llvm::MemoryBuffer::getMemBufferCopy((*BundleBuf)->getBuffer()));

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

  PM.run(*PTU.TheModule);

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
