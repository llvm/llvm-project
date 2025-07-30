//=== OpaqueOffloadLinker - debugable command set for clang-linker-wrapper ===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "Clang.h"
#include "clang/Driver/CommonArgs.h"
#include "llvm/ADT/StringMap.h"    
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/Path.h"
#include "clang/Driver/RocmInstallationDetector.h"
#include "AMDGPU.h"
using namespace clang::driver;
using namespace clang::driver::tools;
using namespace clang;
using namespace llvm::opt;

static const char *getOutputFileName(Compilation &C, StringRef Base,
                                     const char *Postfix,
                                     const char *Extension) {
  const char *OutputFileName;
  if (C.getDriver().isSaveTempsEnabled()) {
    OutputFileName =
        C.getArgs().MakeArgString(Base.str() + Postfix + "." + Extension);
  } else {
    std::string TmpName =
        C.getDriver().GetTemporaryPath(Base.str() + Postfix, Extension);
    OutputFileName = C.addTempFile(C.getArgs().MakeArgString(TmpName));
  }
  return OutputFileName;
}

static void addSubArchsWithTargetID(Compilation &C, const ArgList &Args,
                                    const llvm::Triple &Triple,
                                    SmallVectorImpl<std::string> &subarchs) {
  // process OPT_offload_arch_EQ subarch specification
  ToolChain *TC;
  for (auto itr : C.getDriver().getOffloadArchs(
           C, C.getArgs(), Action::OFK_OpenMP, *TC))
    subarchs.push_back(itr.str());

  // process OPT_Xopenmp_target_EQ subarch specification with march
  for (auto itr : Args.getAllArgValues(options::OPT_Xopenmp_target_EQ)) {
    SmallVector<StringRef> marchs;
    StringRef vstr = StringRef(itr);
    if (vstr.starts_with("-march=") || vstr.starts_with("--march=")) {
      vstr.split('=').second.split(marchs, ',');
      for (auto &march : marchs)
        subarchs.push_back(march.str());
    }
  }
}

/// This is an alternative to LinkerWrapper::ConstructJob.
/// This is called when driver option --opaque-offload-linker is specified.

/// opaque-offload-linker requires heterogeneous objects have bitcode
/// because offload LTO is implemented by merging all offloaded bitcodes
/// and then linking in system bitcode libraries followed by opt and then
/// the GPU backend is called only once for each TargetID.

/// foreach(TargetID) {
///   foreach(input) {
///     1 "unpackage" each .o input to create targetID specific bitcode
///   }
///   2 build-select-link to create a merged bc with corrected attributes.
///   3 llvm-link with -internalize -as-needed with system bitcode libraries.
///   4 opt
///   5 llc
///   6 lld
/// }
/// 7 clang-offload-wrapper to output x.img
/// 8 clang (host) -cc1 -embed x.img -x host.bc -o x.o
/// 9 ld.lld  x.o ... -o linkerwrapper ouput
///
void LinkerWrapper::ConstructOpaqueJob(Compilation &C, const JobAction &JA,
                                       const InputInfo &Output,
                                       const InputInfoList &Inputs,
                                       const ArgList &Args,
                                       const llvm::Triple &TheTriple,
                                       const char *LinkingOutput) const {
  const ToolChain &TC = getToolChain();
  const Driver &D = getToolChain().getDriver();
  RocmInstallationDetector RocmInstallation(D, TheTriple, Args, true, true);
  std::string OutputFilePrefix, OutputFile;

  SmallVector<std::string> subarchs;
  llvm::SmallVector<std::pair<StringRef, const char *>, 4> TargetIDLLDMap;

  addSubArchsWithTargetID(C, Args, TheTriple, subarchs);

  for (auto &subArchWithTargetID : subarchs) {
    StringRef TargetID(subArchWithTargetID);
    // ---------- Step 1 unpackage each input -----------
    const char *UnpackageExec = Args.MakeArgString(
        getToolChain().GetProgramPath("clang-offload-packager"));

    SmallVector<std::string> UnpackagedFiles;

    for (const auto &II : Inputs) {
      if (II.isFilename()) {
        OutputFile = llvm::sys::path::stem(II.getFilename()).str();
        OutputFilePrefix = llvm::sys::path::stem(II.getBaseInput()).str() +
                           "-openmp-" + TheTriple.str() + "-" + TargetID.str();

        // generate command to unpackage each II.getFilename()
        auto UnpackagedFileName =
            getOutputFileName(C, OutputFilePrefix, "-unpackaged", "bc");
        // push unpacked file names to argument list for clang-build-select
        UnpackagedFiles.push_back(UnpackagedFileName);
        ArgStringList UnpackageCmdArgs;
        UnpackageCmdArgs.push_back(II.getFilename());

        SmallVector<std::string> Parts{
            "file=" + std::string(UnpackagedFileName),
            "triple=" + TheTriple.str(),
            "arch=" + TargetID.str(),
            "kind=openmp",
        };

        UnpackageCmdArgs.push_back(
            Args.MakeArgString("--image=" + llvm::join(Parts, ",")));

        UnpackageCmdArgs.push_back("--allow-missing-packages");

        C.addCommand(std::make_unique<Command>(
            JA, *this, ResponseFileSupport::AtFileCurCP(), UnpackageExec,
            UnpackageCmdArgs, Inputs,
            InputInfo(&JA, Args.MakeArgString(UnpackagedFileName))));
      }
    }

    // ---------- Step 2 clang-build-select-link -----------
    // Look for Static Device Libs (SDLs) in args, and add temp files for
    // the extracted Device-specific Archive Libs (DAL) to inputs
    ArgStringList CbslArgs;
    AddStaticDeviceLibsLinking(C, *this, JA, Inputs, Args, CbslArgs, "amdgcn",
                               TargetID,
                               /* bitcode SDL?*/ true,
                               /* PostClang Link? */ false,
                               /* Unpackage? */ true);

    auto PreLinkFileName = amdgpu::dlr::getCbslCommandArgs(
        C, Args, CbslArgs, UnpackagedFiles, OutputFilePrefix);

    const char *CbslExec = Args.MakeArgString(
        getToolChain().GetProgramPath("clang-build-select-link"));
    C.addCommand(std::make_unique<Command>(
        JA, *this, ResponseFileSupport::AtFileCurCP(), CbslExec, CbslArgs,
        Inputs, InputInfo(&JA, Args.MakeArgString(PreLinkFileName))));

    // ---------- Step 3 llvm-link internalize as-needed -----------
    ArgStringList LastLinkArgs;
    // Find all directories pointed to by the environment variable
    // LIBRARY_PATH.
    ArgStringList EnvLibraryPaths;
    addDirectoryList(Args, EnvLibraryPaths, "", "LIBRARY_PATH");
    auto LinkOutputFileName = amdgpu::dlr::getLinkCommandArgs(
        C, Args, LastLinkArgs, TC, TheTriple, TargetID, OutputFilePrefix,
        PreLinkFileName, RocmInstallation, EnvLibraryPaths);

    const char *LinkExec =
        Args.MakeArgString(getToolChain().GetProgramPath("llvm-link"));
    C.addCommand(std::make_unique<Command>(
        JA, *this, ResponseFileSupport::AtFileCurCP(), LinkExec, LastLinkArgs,
        Inputs, InputInfo(&JA, Args.MakeArgString(LinkOutputFileName))));

    // ---------- Step 4 opt  -----------
    ArgStringList OptArgs;

    // Forward -Xopaque-offload-opt arguments to the 'opt' job.
    for (Arg *A : Args.filtered(options::OPT_Xopaque_offload_opt)) {
      OptArgs.push_back(A->getValue());
      A->claim();
    }

    auto OptOutputFileName =
        amdgpu::dlr::getOptCommandArgs(C, Args, OptArgs, TheTriple, TargetID,
                                       OutputFilePrefix, LinkOutputFileName);

    const char *OptExec =
        Args.MakeArgString(getToolChain().GetProgramPath("opt"));
    C.addCommand(std::make_unique<Command>(
        JA, *this, ResponseFileSupport::AtFileCurCP(), OptExec, OptArgs, Inputs,
        InputInfo(&JA, Args.MakeArgString(OptOutputFileName))));

    // ---------- Step 5 llc  -----------
    ArgStringList LlcArgs;
    auto LlcOutputFileName =
        amdgpu::dlr::getLlcCommandArgs(C, Args, LlcArgs, TheTriple, TargetID,
                                       OutputFilePrefix, OptOutputFileName);

    const char *LlcExec =
        Args.MakeArgString(getToolChain().GetProgramPath("llc"));

    // produce assembly temp output file if --save-temps is specified
    if (C.getDriver().isSaveTempsEnabled()) {
      ArgStringList LlcAsmArgs;
      auto LlcAsmOutputFileName = amdgpu::dlr::getLlcCommandArgs(
          C, Args, LlcAsmArgs, TheTriple, TargetID, OutputFilePrefix,
          OptOutputFileName, /*OutputIsAsm*/ true);

      C.addCommand(std::make_unique<Command>(
          JA, *this, ResponseFileSupport::AtFileCurCP(), LlcExec, LlcAsmArgs,
          Inputs, InputInfo(&JA, Args.MakeArgString(LlcAsmOutputFileName))));
    }

    C.addCommand(std::make_unique<Command>(
        JA, *this, ResponseFileSupport::AtFileCurCP(), LlcExec, LlcArgs, Inputs,
        InputInfo(&JA, Args.MakeArgString(LlcOutputFileName))));

    // ---------- Step 6 lld  -----------
    ArgStringList LldArgs;
    auto LldOutputFileName = amdgpu::dlr::getLldCommandArgs(
        C, Output, Args, LldArgs, TheTriple, TargetID, LlcOutputFileName,
        OutputFilePrefix);

    // create vector of pairs of TargetID,lldname for step 7 inputs.
    TargetIDLLDMap.push_back(
        std::pair<StringRef, const char *>(TargetID, LldOutputFileName));

    const char *LldExec =
        Args.MakeArgString(getToolChain().GetProgramPath("lld"));
    C.addCommand(std::make_unique<Command>(
        JA, *this, ResponseFileSupport::AtFileCurCP(), LldExec, LldArgs, Inputs,
        InputInfo(&JA, Args.MakeArgString(LldOutputFileName))));

  } //  End loop for each subarch

  // -------- Step 7 clang-offload-wrapper to build device image
  auto CowOutputFileName = getOutputFileName(C, OutputFile, "-wrapped", "bc");
  ArgStringList CowArgs;
  const char *CowExec = Args.MakeArgString(
      getToolChain().GetProgramPath("clang-offload-wrapper"));

  // The offload target.
  CowArgs.push_back("-target");
  CowArgs.push_back(Args.MakeArgString(TheTriple.getTriple()));

  const llvm::Triple &Triple = getToolChain().getEffectiveTriple();

  // The host triple is the "effective" target triple here.
  CowArgs.push_back("-aux-triple");
  CowArgs.push_back(Args.MakeArgString(Triple.getTriple()));

  // Add the output file name.
  assert(CowOutputFileName != nullptr && "Invalid output.");
  CowArgs.push_back("-o");
  CowArgs.push_back(CowOutputFileName);

  // a vector of pairs of TargetID,lldName
  for (auto &TM : TargetIDLLDMap) {
    CowArgs.push_back(Args.MakeArgString(Twine("--offload-arch=") + TM.first));
    CowArgs.push_back(TM.second);
  }

  C.addCommand(std::make_unique<Command>(
      JA, *this, ResponseFileSupport::AtFileCurCP(), CowExec, CowArgs, Inputs,
      InputInfo(&JA, Args.MakeArgString(CowOutputFileName))));

  // ---------- Step 8 clang -cc1 host backend -----------
  ArgStringList HbeArgs;
  const char *HbeOutputFileName =
      getOutputFileName(C, OutputFilePrefix, "-hbe", "o");
  const char *HbeExec =
      Args.MakeArgString(getToolChain().GetProgramPath("clang"));

  HbeArgs.push_back("-cc1");
  HbeArgs.push_back("-triple");
  HbeArgs.push_back(Args.MakeArgString(getToolChain().getTripleString()));
  HbeArgs.push_back("-emit-obj");
  HbeArgs.push_back("-o");
  HbeArgs.push_back(Args.MakeArgString(HbeOutputFileName));
  HbeArgs.push_back("-x");
  HbeArgs.push_back("ir");
  HbeArgs.push_back(Args.MakeArgString(CowOutputFileName));

  C.addCommand(std::make_unique<Command>(
      JA, *this, ResponseFileSupport::AtFileCurCP(), HbeExec, HbeArgs, Inputs,
      InputInfo(&JA, Args.MakeArgString(HbeOutputFileName))));

  // ---------- Step 9 final host link  -----------
  InputInfoList LinkInputs;
  for (const auto &II : Inputs)
    LinkInputs.push_back(II);

  LinkInputs.push_back(
      InputInfo(types::TY_Object, HbeOutputFileName, HbeOutputFileName));

  Linker->ConstructJob(C, JA, Output, LinkInputs, Args, LinkingOutput);
}
