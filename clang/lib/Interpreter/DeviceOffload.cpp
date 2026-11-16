//===---------- DeviceOffload.cpp - Device Offloading------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//

// This file implements offloading to devices.
//
//===----------------------------------------------------------------------===//

#include "DeviceOffload.h"

#include "clang/Basic/TargetID.h"
#include "clang/Basic/TargetOptions.h"
#include "clang/CodeGen/BackendUtil.h"
#include "clang/CodeGen/ModuleBuilder.h"
#include "clang/Driver/CommonArgs.h"
#include "clang/Driver/OffloadBundler.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Interpreter/PartialTranslationUnit.h"
#include "clang/Options/Options.h"

#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Module.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Option/ArgList.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/FileUtilities.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/TargetParser/AMDGPUTargetParser.h"
#include "llvm/TargetParser/Host.h"
#include "llvm/TargetParser/Triple.h"

namespace clang {

IncrementalDeviceParser::IncrementalDeviceParser(
    CompilerInstance &DeviceInstance, CompilerInstance &HostInstance,
    IncrementalAction *DeviceAct,
    llvm::IntrusiveRefCntPtr<llvm::vfs::InMemoryFileSystem> FS,
    llvm::Error &Err, std::list<PartialTranslationUnit> &PTUs)
    : IncrementalParser(DeviceInstance, DeviceAct, Err, PTUs), VFS(FS),
      CodeGenOpts(HostInstance.getCodeGenOpts()),
      TargetOpts(DeviceInstance.getTargetOpts()) {}

IncrementalDeviceParser::~IncrementalDeviceParser() {}

IncrementalHIPDeviceParser::IncrementalHIPDeviceParser(
    CompilerInstance &DeviceInstance, CompilerInstance &HostInstance,
    IncrementalAction *DeviceAct,
    llvm::IntrusiveRefCntPtr<llvm::vfs::InMemoryFileSystem> FS,
    llvm::Error &Err, std::list<PartialTranslationUnit> &PTUs,
    llvm::ArrayRef<const char *> DriverArgs)
    : IncrementalDeviceParser(DeviceInstance, HostInstance, DeviceAct, FS, Err,
                              PTUs),
      DeviceCI(DeviceInstance) {
  if (Err)
    return;
  if (!llvm::Triple(TargetOpts.Triple).isAMDGCN()) {
    Err = llvm::joinErrors(
        std::move(Err),
        llvm::make_error<llvm::StringError>(
            "Invalid HIP architecture for target '" + TargetOpts.Triple + "'",
            llvm::inconvertibleErrorCode()));
    return;
  }

  unsigned MissingArgIndex, MissingArgCount;
  llvm::opt::InputArgList Args = getDriverOptTable().ParseArgs(
      DriverArgs, MissingArgIndex, MissingArgCount,
      llvm::opt::Visibility(options::ClangOption));
  driver::tools::addPrefixDirs(Args, PrefixDirs);
  LdPath = Args.getLastArgValue(options::OPT_ld_path_EQ).str();
}

static std::string getProgramPath(llvm::StringRef Name,
                                  llvm::ArrayRef<std::string> PrefixDirs) {
  if (std::optional<std::string> P =
          driver::tools::findProgramInPrefixDirs(Name, PrefixDirs))
    return *P;
  std::string Exe = llvm::sys::fs::getMainExecutable(nullptr, nullptr);
  if (llvm::ErrorOr<std::string> P = llvm::sys::findProgramByName(
          Name, {llvm::sys::path::parent_path(Exe)}))
    return *P;
  if (llvm::ErrorOr<std::string> P = llvm::sys::findProgramByName(Name))
    return *P;
  return Name.str();
}

llvm::Expected<llvm::StringRef>
IncrementalHIPDeviceParser::GenerateDeviceObject() {
  auto &PTU = PTUs.back();

  CodeGenOptions CodeGenOptsForObj = DeviceCI.getCodeGenOpts();
  CodeGenOptsForObj.DisableLLVMPasses = true;

  ObjectContent.clear();
  auto ObjOS = std::make_unique<llvm::raw_svector_ostream>(ObjectContent);
  emitBackendOutput(DeviceCI, CodeGenOptsForObj, PTU.TheModule.get(),
                    Backend_EmitObj, DeviceCI.getVirtualFileSystemPtr(),
                    std::move(ObjOS));

  if (DeviceCI.getDiagnostics().hasErrorOccurred())
    return llvm::make_error<llvm::StringError>(
        "Backend code generation failed for HIP device code.",
        llvm::inconvertibleErrorCode());

  return llvm::StringRef(ObjectContent.data(), ObjectContent.size());
}

llvm::Expected<llvm::StringRef>
IncrementalHIPDeviceParser::GenerateHSACO(llvm::StringRef Object) {
  // Without --ld-path, look for a program named "lld".
  llvm::StringRef LinkerName = LdPath;
  if (LinkerName.empty())
    LinkerName = "lld";
  std::optional<std::string> Linker =
      driver::tools::resolveLdPath(LinkerName, [&](llvm::StringRef Name) {
        return getProgramPath(Name, PrefixDirs);
      });
  if (!Linker)
    return llvm::make_error<llvm::StringError>(
        LdPath.empty()
            ? "Could not find lld."
            : "Invalid linker name in argument '--ld-path=" + LdPath + "'.",
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
    OS << Object;
  }

  llvm::SmallString<128> HSACOFile;
  if (llvm::sys::fs::createTemporaryFile("kernel", "hsaco", HSACOFile))
    return llvm::make_error<llvm::StringError>(
        "Failed to create a temporary code object file.",
        llvm::inconvertibleErrorCode());
  llvm::FileRemover HSACORemover(HSACOFile);

  // These match the driver's HIP device link (AMDGCN::Linker) for an object.
  llvm::StringRef Args[] = {*Linker,   "-flavor",         "gnu",
                            "-m",      "elf64_amdgpu",    "--no-undefined",
                            "-shared", "--whole-archive", "-o",
                            HSACOFile, ObjFile,           "--no-whole-archive"};
  if (llvm::sys::ExecuteAndWait(*Linker, Args) != 0)
    return llvm::make_error<llvm::StringError>("The linker invocation failed.",
                                               llvm::inconvertibleErrorCode());

  auto HSACOBuf = llvm::MemoryBuffer::getFile(HSACOFile, /*IsText=*/false);
  if (!HSACOBuf)
    return llvm::make_error<llvm::StringError>(
        "Failed to read the code object.", llvm::inconvertibleErrorCode());

  llvm::StringRef Buffer = (*HSACOBuf)->getBuffer();
  HSACOContent.assign(Buffer.begin(), Buffer.end());
  return llvm::StringRef(HSACOContent.data(), HSACOContent.size());
}

llvm::Expected<llvm::StringRef>
IncrementalHIPDeviceParser::GenerateFatbinary(llvm::StringRef HSACO) {
  int HSACOFD = -1;
  llvm::SmallString<128> HSACOFile;
  if (llvm::sys::fs::createTemporaryFile("kernel", "hsaco", HSACOFD, HSACOFile))
    return llvm::make_error<llvm::StringError>(
        "Failed to create a temporary code object file.",
        llvm::inconvertibleErrorCode());
  llvm::FileRemover HSACORemover(HSACOFile);
  {
    llvm::raw_fd_ostream OS(HSACOFD, /*shouldClose=*/true);
    OS << HSACO;
  }

  llvm::StringRef OffloadKind =
      (TargetOpts.CodeObjectVersion == llvm::CodeObjectVersionKind::COV_2 ||
       TargetOpts.CodeObjectVersion == llvm::CodeObjectVersionKind::COV_3)
          ? "hip"
          : "hipv4";
  llvm::SmallString<128> FatbinFile;
  if (llvm::sys::fs::createTemporaryFile("kernel", "hipfb", FatbinFile))
    return llvm::make_error<llvm::StringError>(
        "Failed to create a temporary fat binary file.",
        llvm::inconvertibleErrorCode());
  llvm::FileRemover FatbinRemover(FatbinFile);

  llvm::SmallString<128> HostFile;
  if (llvm::sys::fs::createTemporaryFile("host", "o", HostFile))
    return llvm::make_error<llvm::StringError>(
        "Failed to create a temporary host file.",
        llvm::inconvertibleErrorCode());
  llvm::FileRemover HostRemover(HostFile);

  llvm::Triple DeviceTriple(TargetOpts.Triple);
  // The GPU can be encoded in the triple's sub-architecture instead of in
  // -target-cpu, so fall back to it when the CPU is empty.
  std::string Processor = TargetOpts.CPU;
  if (Processor.empty())
    if (std::optional<llvm::AMDGPU::TargetID> ID =
            llvm::AMDGPU::TargetID::parse(DeviceTriple, ""))
      Processor = ID->getCanonicalFeatureString();
  std::string TargetID =
      llvm::AMDGPU::TargetID::createFromSubtargetFeatures(
          DeviceTriple, Processor, llvm::join(TargetOpts.Features, ","))
          .getCanonicalFeatureString();
  if (TargetID.empty())
    return llvm::make_error<llvm::StringError>(
        "Could not determine the HIP device architecture.",
        llvm::inconvertibleErrorCode());

  OffloadBundlerConfig BundlerConfig;
  BundlerConfig.BundleAlignment = 4096;
  BundlerConfig.HostInputIndex = 0;
  BundlerConfig.FilesType = "o";
  BundlerConfig.TargetNames = {"host-" + llvm::sys::getProcessTriple(),
                               OffloadKind.str() + "-" +
                                   normalizeForBundler(DeviceTriple, TargetID) +
                                   "-" + TargetID};
  BundlerConfig.InputFileNames = {std::string(HostFile),
                                  std::string(HSACOFile)};
  BundlerConfig.OutputFileNames = {std::string(FatbinFile)};

  if (llvm::Error Err = OffloadBundler(BundlerConfig).BundleFiles())
    return std::move(Err);

  auto FatbinBuf = llvm::MemoryBuffer::getFile(FatbinFile, /*IsText=*/false);
  if (!FatbinBuf)
    return llvm::make_error<llvm::StringError>("Failed to read the fat binary.",
                                               llvm::inconvertibleErrorCode());

  llvm::StringRef Buffer = (*FatbinBuf)->getBuffer();
  FatbinContent.assign(Buffer.begin(), Buffer.end());
  return llvm::StringRef(FatbinContent.data(), FatbinContent.size());
}

void IncrementalHIPDeviceParser::EmbedOffloadBundle() {
  const PartialTranslationUnit &PTU = PTUs.back();
  std::string BundleFileName = "/" + PTU.TheModule->getName().str() + ".hipfb";
  VFS->addFile(BundleFileName, 0,
               llvm::MemoryBuffer::getMemBufferCopy(llvm::StringRef(
                   FatbinContent.data(), FatbinContent.size())));

  CodeGenOpts.OffloadBinaryToEmbedFile = std::move(BundleFileName);
}

llvm::Error IncrementalHIPDeviceParser::GenerateOffloadBinary() {
  llvm::Expected<llvm::StringRef> Object = GenerateDeviceObject();
  if (!Object)
    return Object.takeError();

  llvm::Expected<llvm::StringRef> HSACO = GenerateHSACO(*Object);
  if (!HSACO)
    return HSACO.takeError();

  llvm::Expected<llvm::StringRef> Fatbin = GenerateFatbinary(*HSACO);
  if (!Fatbin)
    return Fatbin.takeError();

  EmbedOffloadBundle();
  return llvm::Error::success();
}

IncrementalHIPDeviceParser::~IncrementalHIPDeviceParser() {}

IncrementalCUDADeviceParser::IncrementalCUDADeviceParser(
    CompilerInstance &DeviceInstance, CompilerInstance &HostInstance,
    IncrementalAction *DeviceAct,
    llvm::IntrusiveRefCntPtr<llvm::vfs::InMemoryFileSystem> FS,
    llvm::Error &Err, std::list<PartialTranslationUnit> &PTUs)
    : IncrementalDeviceParser(DeviceInstance, HostInstance, DeviceAct, FS, Err,
                              PTUs) {
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
  std::string Error;

  const llvm::Target *Target = llvm::TargetRegistry::lookupTarget(
      PTU.TheModule->getTargetTriple(), Error);
  if (!Target)
    return llvm::make_error<llvm::StringError>(std::move(Error),
                                               std::error_code());
  llvm::TargetOptions TO = llvm::TargetOptions();
  std::unique_ptr<llvm::TargetMachine> TM(Target->createTargetMachine(
      PTU.TheModule->getTargetTriple(), TargetOpts.CPU,
      llvm::join(TargetOpts.Features, ","), TO, llvm::Reloc::Model::PIC_));

  PTXCode.clear();
  llvm::raw_svector_ostream dest(PTXCode);

  llvm::legacy::PassManager PM;
  if (TM->addPassesToEmitFile(PM, dest, nullptr,
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

llvm::Error IncrementalCUDADeviceParser::GenerateOffloadBinary() {
  llvm::Expected<llvm::StringRef> PTX = GeneratePTX();
  if (!PTX)
    return PTX.takeError();

  return GenerateFatbinary();
}

IncrementalCUDADeviceParser::~IncrementalCUDADeviceParser() {}

} // namespace clang
