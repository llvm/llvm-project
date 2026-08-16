//===--- HIPAMD.cpp - HIP Tool and ToolChain Implementations ----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "HIPAMD.h"
#include "AMDGPU.h"
#include "HIPUtility.h"
#include "SPIRV.h"
#include "clang/Basic/Cuda.h"
#include "clang/Driver/CommonArgs.h"
#include "clang/Driver/Compilation.h"
#include "clang/Driver/Driver.h"
#include "clang/Driver/InputInfo.h"
#include "clang/Driver/SanitizerArgs.h"
#include "clang/Options/Options.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/TargetParser/TargetParser.h"

using namespace clang::driver;
using namespace clang::driver::toolchains;
using namespace clang::driver::tools;
using namespace clang;
using namespace llvm::opt;

#if defined(_WIN32) || defined(_WIN64)
#define NULL_FILE "nul"
#else
#define NULL_FILE "/dev/null"
#endif

void AMDGCN::Linker::constructLLVMLinkCommand(
    Compilation &C, const JobAction &JA, const InputInfoList &Inputs,
    const InputInfo &Output, const llvm::opt::ArgList &Args) const {

  ArgStringList LinkerInputs;

  for (auto Input : Inputs)
    if (Input.isFilename())
      LinkerInputs.push_back(Input.getFilename());

  // Look for archive of bundled bitcode in arguments, and add temporary files
  // for the extracted archive of bitcode to inputs.
  auto TargetID = Args.getLastArgValue(options::OPT_mcpu_EQ);
  AddStaticDeviceLibsLinking(C, *this, JA, Inputs, Args, LinkerInputs, "amdgcn",
                             TargetID, /*IsBitCodeSDL=*/true);
  tools::constructLLVMLinkCommand(C, *this, JA, Inputs, LinkerInputs, Output,
                                  Args);
}

void AMDGCN::Linker::constructLldCommand(Compilation &C, const JobAction &JA,
                                         const InputInfoList &Inputs,
                                         const InputInfo &Output,
                                         const llvm::opt::ArgList &Args) const {
  // Construct lld command.
  // The output from ld.lld is an HSA code object file.
  ArgStringList LldArgs{"-flavor",
                        "gnu",
                        "-m",
                        "elf64_amdgpu",
                        "--no-undefined",
                        "-shared",
                        "-plugin-opt=-amdgpu-internalize-symbols"};
  if (Args.hasArg(options::OPT_hipstdpar))
    LldArgs.push_back("-plugin-opt=-amdgpu-enable-hipstdpar");

  auto &TC = getToolChain();
  auto &D = TC.getDriver();
  bool IsThinLTO = TC.getLTOMode(Args, Action::OFK_HIP) == LTOK_Thin;
  addLTOOptions(TC, Args, LldArgs, Output, Inputs, IsThinLTO);

  // Extract all the -m options
  std::vector<llvm::StringRef> Features;
  amdgpu::getAMDGPUTargetFeatures(D, TC.getEffectiveTriple(), Args, Features);

  // Add features to mattr such as cumode
  std::string MAttrString = "-plugin-opt=-mattr=";
  for (auto OneFeature : unifyTargetFeatures(Features)) {
    MAttrString.append(Args.MakeArgStringRef(OneFeature));
    if (OneFeature != Features.back())
      MAttrString.append(",");
  }
  if (!Features.empty())
    LldArgs.push_back(Args.MakeArgString(MAttrString));

  // ToDo: Remove this option after AMDGPU backend supports ISA-level linking.
  // Since AMDGPU backend currently does not support ISA-level linking, all
  // called functions need to be imported.
  if (IsThinLTO) {
    LldArgs.push_back("-plugin-opt=-force-import-all");
    LldArgs.push_back("-plugin-opt=-avail-extern-to-local");
    LldArgs.push_back("-plugin-opt=-avail-extern-gv-in-addrspace-to-local=3");
  }

  for (const Arg *A : Args.filtered(options::OPT_mllvm)) {
    LldArgs.push_back(
        Args.MakeArgString(Twine("-plugin-opt=") + A->getValue(0)));
  }

  if (C.getDriver().isSaveTempsEnabled())
    LldArgs.push_back("-save-temps");

  addLinkerCompressDebugSectionsOption(TC, Args, LldArgs);

  // Given that host and device linking happen in separate processes, the device
  // linker doesn't always have the visibility as to which device symbols are
  // needed by a program, especially for the device symbol dependencies that are
  // introduced through the host symbol resolution.
  // For example: host_A() (A.obj) --> host_B(B.obj) --> device_kernel_B()
  // (B.obj) In this case, the device linker doesn't know that A.obj actually
  // depends on the kernel functions in B.obj.  When linking to static device
  // library, the device linker may drop some of the device global symbols if
  // they aren't referenced.  As a workaround, we are adding to the
  // --whole-archive flag such that all global symbols would be linked in.
  LldArgs.push_back("--whole-archive");

  for (auto *Arg : Args.filtered(options::OPT_Xoffload_linker)) {
    StringRef ArgVal = Arg->getValue(1);
    auto SplitArg = ArgVal.split("-mllvm=");
    if (!SplitArg.second.empty()) {
      LldArgs.push_back(
          Args.MakeArgString(Twine("-plugin-opt=") + SplitArg.second));
    } else {
      LldArgs.push_back(Args.MakeArgStringRef(ArgVal));
    }
    Arg->claim();
  }

  LldArgs.append({"-o", Output.getFilename()});
  for (auto Input : Inputs)
    LldArgs.push_back(Input.getFilename());

  // Look for archive of bundled bitcode in arguments, and add temporary files
  // for the extracted archive of bitcode to inputs.
  auto TargetID = Args.getLastArgValue(options::OPT_mcpu_EQ);
  AddStaticDeviceLibsLinking(C, *this, JA, Inputs, Args, LldArgs, "amdgcn",
                             TargetID, /*IsBitCodeSDL=*/true);

  LldArgs.push_back("--no-whole-archive");

  const char *Lld = Args.MakeArgStringRef(getToolChain().GetProgramPath("lld"));
  C.addCommand(std::make_unique<Command>(JA, *this, ResponseFileSupport::None(),
                                         Lld, LldArgs, Inputs, Output));
}

// For SPIR-V the inputs for the job are device AMDGCN SPIR-V flavoured bitcode
// and the output is either a compiled SPIR-V binary or bitcode (-emit-llvm). It
// calls llvm-link and then the llvm-spirv translator or the SPIR-V BE.
// TODO: consider if we want to run any targeted optimisations over IR here,
// over generic SPIR-V.
void AMDGCN::Linker::constructLinkAndEmitSpirvCommand(
    Compilation &C, const JobAction &JA, const InputInfoList &Inputs,
    const InputInfo &Output, const llvm::opt::ArgList &Args) const {
  assert(!Inputs.empty() && "Must have at least one input.");

  std::string LinkedBCFilePrefix(
      Twine(llvm::sys::path::stem(Output.getFilename()), "-linked").str());
  const char *LinkedBCFilePath = HIP::getTempFile(C, LinkedBCFilePrefix, "bc");
  InputInfo LinkedBCFile(&JA, LinkedBCFilePath, Output.getBaseInput());

  bool UseSPIRVBackend = Args.hasFlag(options::OPT_use_spirv_backend,
                                      options::OPT_no_use_spirv_backend,
                                      /*Default=*/true);

  constructLLVMLinkCommand(C, JA, Inputs, LinkedBCFile, Args);

  if (UseSPIRVBackend) {
    // This code handles the case in the new driver when --offload-device-only
    // is unset and clang-linker-wrapper forwards the bitcode that must be
    // compiled to SPIR-V.

    llvm::opt::ArgStringList CmdArgs;

    CmdArgs.append({"-cc1", "-triple=spirv64-amd-amdhsa", "-emit-obj",
                    "-disable-llvm-optzns", "-mllvm", "-spirv-preserve-auxdata",
                    LinkedBCFile.getFilename(), "-o", Output.getFilename()});

    const Driver &Driver = getToolChain().getDriver();
    const char *Exec = Driver.getDriverProgramPath();
    C.addCommand(std::make_unique<Command>(
        JA, *this, ResponseFileSupport::None(), Exec, CmdArgs, LinkedBCFile,
        Output, Driver.getPrependArg()));
  } else {
    // Emit SPIR-V binary using the translator
    llvm::opt::ArgStringList TrArgs{
        "--spirv-max-version=1.6",
        "--spirv-ext=+all",
        "--spirv-allow-unknown-intrinsics",
        "--spirv-lower-const-expr",
        "--spirv-preserve-auxdata",
        "--spirv-debug-info-version=nonsemantic-shader-200"};
    SPIRV::constructTranslateCommand(C, *this, JA, Output, LinkedBCFile,
                                     TrArgs);
  }
}

// For amdgcn the inputs of the linker job are device bitcode and output is
// either an object file or bitcode (-emit-llvm). It calls llvm-link, opt,
// llc, then lld steps.
void AMDGCN::Linker::ConstructJob(Compilation &C, const JobAction &JA,
                                  const InputInfo &Output,
                                  const InputInfoList &Inputs,
                                  const ArgList &Args,
                                  const char *LinkingOutput) const {
  if (!Inputs.empty() && Inputs[0].getType() == types::TY_Image &&
      JA.getType() == types::TY_Object)
    return HIP::constructGenerateObjFileFromHIPFatBinary(C, Output, Inputs,
                                                         Args, JA, *this);

  if (JA.getType() == types::TY_HIP_FATBIN)
    return HIP::constructHIPFatbinCommand(C, JA, Output.getFilename(), Inputs,
                                          Args, *this);

  if (JA.getType() == types::TY_LLVM_BC)
    return constructLLVMLinkCommand(C, JA, Inputs, Output, Args);

  if (getToolChain().getEffectiveTriple().isSPIRV())
    return constructLinkAndEmitSpirvCommand(C, JA, Inputs, Output, Args);

  return constructLldCommand(C, JA, Inputs, Output, Args);
}

SPIRVAMDToolChain::SPIRVAMDToolChain(const Driver &D,
                                     const llvm::Triple &Triple,
                                     const llvm::opt::ArgList &Args)
    : AMDGPUToolChain(D, Triple, Args) {}

Tool *SPIRVAMDToolChain::buildLinker() const {
  return new tools::AMDGCN::Linker(*this);
}
