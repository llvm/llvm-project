//===--- HIPUtility.cpp - Common HIP Tool Chain Utilities -------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "HIPUtility.h"
#include "clang/Driver/CommonArgs.h"
#include "clang/Driver/Compilation.h"
#include "clang/Driver/OffloadBundler.h"
#include "clang/Options/Options.h"
#include "llvm/Support/Path.h"

using namespace clang;
using namespace clang::driver;
using namespace clang::driver::tools;
using namespace llvm::opt;

#if defined(_WIN32) || defined(_WIN64)
#define NULL_FILE "nul"
#else
#define NULL_FILE "/dev/null"
#endif

namespace {
const unsigned HIPCodeObjectAlign = 4096;
} // namespace

// Collect undefined __hip_fatbin* and __hip_gpubin_handle* symbols from all
// input object or archive files.
class HIPUndefinedFatBinSymbols {
public:
  HIPUndefinedFatBinSymbols(const Compilation &C,
                            const llvm::opt::ArgList &Args_)
      : C(C), Args(Args_),
        DiagID(C.getDriver().getDiags().getCustomDiagID(
            DiagnosticsEngine::Error,
            "Error collecting HIP undefined fatbin symbols: %0")),
        Quiet(C.getArgs().hasArg(options::OPT__HASH_HASH_HASH)),
        Verbose(C.getArgs().hasArg(options::OPT_v)) {
    populateSymbols();
    processStaticLibraries();
    if (Verbose) {
      for (const auto &Name : FatBinSymbols)
        llvm::errs() << "Found undefined HIP fatbin symbol: " << Name << "\n";
      for (const auto &Name : GPUBinHandleSymbols)
        llvm::errs() << "Found undefined HIP gpubin handle symbol: " << Name
                     << "\n";
    }
  }

  const std::set<std::string> &getFatBinSymbols() const {
    return FatBinSymbols;
  }

  const std::set<std::string> &getGPUBinHandleSymbols() const {
    return GPUBinHandleSymbols;
  }

  // Collect symbols from static libraries specified by -l options.
  void processStaticLibraries() {
    llvm::SmallVector<llvm::StringRef, 16> LibNames;
    llvm::SmallVector<llvm::StringRef, 16> LibPaths;
    llvm::SmallVector<llvm::StringRef, 16> ExactLibNames;
    llvm::Triple Triple(C.getDriver().getTargetTriple());
    bool IsMSVC = Triple.isWindowsMSVCEnvironment();
    llvm::StringRef Ext = IsMSVC ? ".lib" : ".a";

    for (const auto *Arg : Args.filtered(options::OPT_l)) {
      llvm::StringRef Value = Arg->getValue();
      if (Value.starts_with(":"))
        ExactLibNames.push_back(Value.drop_front());
      else
        LibNames.push_back(Value);
    }
    for (const auto *Arg : Args.filtered(options::OPT_L)) {
      auto Path = Arg->getValue();
      LibPaths.push_back(Path);
      if (Verbose)
        llvm::errs() << "HIP fatbin symbol search uses library path:  " << Path
                     << "\n";
    }

    auto ProcessLib = [&](llvm::StringRef LibName, bool IsExact) {
      llvm::SmallString<256> FullLibName(
          IsExact  ? Twine(LibName).str()
          : IsMSVC ? (Twine(LibName) + Ext).str()
                   : (Twine("lib") + LibName + Ext).str());

      bool Found = false;
      for (const auto Path : LibPaths) {
        llvm::SmallString<256> FullPath = Path;
        llvm::sys::path::append(FullPath, FullLibName);

        if (llvm::sys::fs::exists(FullPath)) {
          if (Verbose)
            llvm::errs() << "HIP fatbin symbol search found library: "
                         << FullPath << "\n";
          auto BufferOrErr = llvm::MemoryBuffer::getFile(FullPath);
          if (!BufferOrErr) {
            errorHandler(llvm::errorCodeToError(BufferOrErr.getError()));
            continue;
          }
          processInput(BufferOrErr.get()->getMemBufferRef());
          Found = true;
          break;
        }
      }
      if (!Found && Verbose)
        llvm::errs() << "HIP fatbin symbol search could not find library: "
                     << FullLibName << "\n";
    };

    for (const auto LibName : ExactLibNames)
      ProcessLib(LibName, true);

    for (const auto LibName : LibNames)
      ProcessLib(LibName, false);
  }

private:
  const Compilation &C;
  const llvm::opt::ArgList &Args;
  unsigned DiagID;
  bool Quiet;
  bool Verbose;
  std::set<std::string> FatBinSymbols;
  std::set<std::string> GPUBinHandleSymbols;
  std::set<std::string, std::less<>> DefinedFatBinSymbols;
  std::set<std::string, std::less<>> DefinedGPUBinHandleSymbols;
  const std::string FatBinPrefix = "__hip_fatbin";
  const std::string GPUBinHandlePrefix = "__hip_gpubin_handle";

  void populateSymbols() {
    std::deque<const Action *> WorkList;
    std::set<const Action *> Visited;

    for (const auto &Action : C.getActions())
      WorkList.push_back(Action);

    while (!WorkList.empty()) {
      const Action *CurrentAction = WorkList.front();
      WorkList.pop_front();

      if (!CurrentAction || !Visited.insert(CurrentAction).second)
        continue;

      if (const auto *IA = dyn_cast<InputAction>(CurrentAction)) {
        std::string ID = IA->getId().str();
        if (!ID.empty()) {
          ID = llvm::utohexstr(llvm::MD5Hash(ID), /*LowerCase=*/true);
          FatBinSymbols.insert((FatBinPrefix + Twine('_') + ID).str());
          GPUBinHandleSymbols.insert(
              (GPUBinHandlePrefix + Twine('_') + ID).str());
          continue;
        }
        if (IA->getInputArg().getNumValues() == 0)
          continue;
        const char *Filename = IA->getInputArg().getValue();
        if (!Filename)
          continue;
        auto BufferOrErr = llvm::MemoryBuffer::getFile(Filename);
        // Input action could be options to linker, therefore, ignore it
        // if cannot read it. If it turns out to be a file that cannot be read,
        // the error will be caught by the linker.
        if (!BufferOrErr)
          continue;

        processInput(BufferOrErr.get()->getMemBufferRef());
      } else
        llvm::append_range(WorkList, CurrentAction->getInputs());
    }
  }

  void processInput(const llvm::MemoryBufferRef &Buffer) {
    // Try processing as object file first.
    auto ObjFileOrErr = llvm::object::ObjectFile::createObjectFile(Buffer);
    if (ObjFileOrErr) {
      processSymbols(**ObjFileOrErr);
      return;
    }

    // Then try processing as archive files.
    llvm::consumeError(ObjFileOrErr.takeError());
    auto ArchiveOrErr = llvm::object::Archive::create(Buffer);
    if (ArchiveOrErr) {
      llvm::Error Err = llvm::Error::success();
      llvm::object::Archive &Archive = *ArchiveOrErr.get();
      for (auto &Child : Archive.children(Err)) {
        auto ChildBufOrErr = Child.getMemoryBufferRef();
        if (ChildBufOrErr)
          processInput(*ChildBufOrErr);
        else
          errorHandler(ChildBufOrErr.takeError());
      }

      if (Err)
        errorHandler(std::move(Err));
      return;
    }

    // Ignore other files.
    llvm::consumeError(ArchiveOrErr.takeError());
  }

  void processSymbols(const llvm::object::ObjectFile &Obj) {
    for (const auto &Symbol : Obj.symbols()) {
      auto FlagOrErr = Symbol.getFlags();
      if (!FlagOrErr) {
        errorHandler(FlagOrErr.takeError());
        continue;
      }

      auto NameOrErr = Symbol.getName();
      if (!NameOrErr) {
        errorHandler(NameOrErr.takeError());
        continue;
      }
      llvm::StringRef Name = *NameOrErr;

      bool isUndefined =
          FlagOrErr.get() & llvm::object::SymbolRef::SF_Undefined;
      bool isHidden = FlagOrErr.get() & llvm::object::SymbolRef::SF_Hidden;
      bool isFatBinSymbol = Name.starts_with(FatBinPrefix);
      bool isGPUBinHandleSymbol = Name.starts_with(GPUBinHandlePrefix);

      // Add undefined symbols if they are not in the defined sets
      if (isUndefined) {
        if (isFatBinSymbol &&
            DefinedFatBinSymbols.find(Name) == DefinedFatBinSymbols.end())
          FatBinSymbols.insert(Name.str());
        else if (isGPUBinHandleSymbol &&
                 DefinedGPUBinHandleSymbols.find(Name) ==
                     DefinedGPUBinHandleSymbols.end())
          GPUBinHandleSymbols.insert(Name.str());
        continue;
      }

      // Ignore hidden defined symbols
      if (isHidden)
        continue;

      // Handling for non-hidden defined symbols
      if (isFatBinSymbol) {
        DefinedFatBinSymbols.insert(Name.str());
        FatBinSymbols.erase(Name.str());
      } else if (isGPUBinHandleSymbol) {
        DefinedGPUBinHandleSymbols.insert(Name.str());
        GPUBinHandleSymbols.erase(Name.str());
      }
    }
  }

  void errorHandler(llvm::Error Err) {
    if (Quiet)
      return;
    C.getDriver().Diag(DiagID) << llvm::toString(std::move(Err));
  }
};

// Construct a clang-offload-bundler command to bundle code objects for
// different devices into a HIP fat binary.
void HIP::constructHIPFatbinCommand(Compilation &C, const JobAction &JA,
                                    llvm::StringRef OutputFileName,
                                    const InputInfoList &Inputs,
                                    const llvm::opt::ArgList &Args,
                                    const Tool &T) {
  // Construct clang-offload-bundler command to bundle object files for
  // for different GPU archs.
  ArgStringList BundlerArgs;
  BundlerArgs.push_back(Args.MakeArgString("-type=o"));
  BundlerArgs.push_back(
      Args.MakeArgString("-bundle-align=" + Twine(HIPCodeObjectAlign)));

  // ToDo: Remove the dummy host binary entry which is required by
  // clang-offload-bundler.
  std::string BundlerTargetArg = "-targets=host-x86_64-unknown-linux-gnu";
  // AMDGCN:
  // For code object version 2 and 3, the offload kind in bundle ID is 'hip'
  // for backward compatibility. For code object version 4 and greater, the
  // offload kind in bundle ID is 'hipv4'.
  std::string OffloadKind = "hip";
  if (T.getToolChain().getTriple().isAMDGCN() &&
      getAMDGPUCodeObjectVersion(C.getDriver(), Args) >= 4)
    OffloadKind = OffloadKind + "v4";
  for (const auto &II : Inputs) {
    const auto *A = II.getAction();
    const llvm::Triple &InputTriple = A->getOffloadingToolChain()->getTriple();

    BoundArch BA = A->getOffloadingArch();
    BundlerTargetArg += ',' + OffloadKind + '-';
    if (BA.ArchName == "amdgcnspirv")
      BundlerTargetArg += "spirv64-amd-amdhsa-";
    else
      BundlerTargetArg += normalizeForBundler(InputTriple, BA.ArchName);
    if (BA)
      BundlerTargetArg += '-' + BA.ArchName.str();
  }
  BundlerArgs.push_back(Args.MakeArgString(BundlerTargetArg));

  // Use a NULL file as input for the dummy host binary entry
  std::string BundlerInputArg = "-input=" NULL_FILE;
  BundlerArgs.push_back(Args.MakeArgString(BundlerInputArg));
  for (const auto &II : Inputs) {
    BundlerInputArg = std::string("-input=") + II.getFilename();
    BundlerArgs.push_back(Args.MakeArgString(BundlerInputArg));
  }

  std::string Output = std::string(OutputFileName);
  auto *BundlerOutputArg =
      Args.MakeArgString(std::string("-output=").append(Output));
  BundlerArgs.push_back(BundlerOutputArg);

  addOffloadCompressArgs(Args, BundlerArgs);

  const char *Bundler = Args.MakeArgString(
      T.getToolChain().GetProgramPath("clang-offload-bundler"));
  C.addCommand(std::make_unique<Command>(
      JA, T, ResponseFileSupport::None(), Bundler, BundlerArgs, Inputs,
      InputInfo(&JA, Args.MakeArgString(Output))));
}

// Convenience function for creating temporary file for both modes of
// isSaveTempsEnabled().
const char *HIP::getTempFile(Compilation &C, StringRef Prefix,
                             StringRef Extension) {
  if (C.getDriver().isSaveTempsEnabled()) {
    return C.getArgs().MakeArgString(Prefix + "." + Extension);
  }
  auto TmpFile = C.getDriver().GetTemporaryPath(Prefix, Extension);
  return C.addTempFile(C.getArgs().MakeArgString(TmpFile));
}
