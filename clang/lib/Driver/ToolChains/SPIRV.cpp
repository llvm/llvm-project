//===--- SPIRV.cpp - SPIR-V Tool Implementations ----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "SPIRV.h"
#include "HIPUtility.h"
#include "clang/Driver/CommonArgs.h"
#include "clang/Driver/Compilation.h"
#include "clang/Driver/Driver.h"
#include "clang/Driver/InputInfo.h"
#include "clang/Options/Options.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"

using namespace clang::driver;
using namespace clang::driver::toolchains;
using namespace clang::driver::tools;
using namespace llvm::opt;
using namespace clang;

void SPIRV::constructTranslateCommand(Compilation &C, const Tool &T,
                                      const JobAction &JA,
                                      const InputInfo &Output,
                                      const InputInfo &Input,
                                      const llvm::opt::ArgStringList &Args) {
  llvm::opt::ArgStringList CmdArgs(Args);
  CmdArgs.push_back(Input.getFilename());

  assert(Input.getType() != types::TY_PP_Asm && "Unexpected input type");

  if (Output.getType() == types::TY_PP_Asm)
    CmdArgs.push_back("--spirv-tools-dis");

  CmdArgs.append({"-o", Output.getFilename()});

  // Derive llvm-spirv path from clang path to ensure we use the same LLVM version.
  // Try versioned tool first, then fall back to unversioned.
  std::string TranslateCmdClangPath = C.getDriver().getClangProgramPath();
  SmallString<128> TranslateCmdPath(TranslateCmdClangPath);
  llvm::sys::path::remove_filename(TranslateCmdPath);
  SmallString<128> TranslateCmdVersionedPath(TranslateCmdPath);
  llvm::sys::path::append(TranslateCmdVersionedPath, "llvm-spirv-" + std::to_string(LLVM_VERSION_MAJOR));
  if (llvm::sys::fs::can_execute(TranslateCmdVersionedPath)) {
    llvm::sys::path::append(TranslateCmdPath, "llvm-spirv-" + std::to_string(LLVM_VERSION_MAJOR));
  } else {
    llvm::sys::path::append(TranslateCmdPath, "llvm-spirv");
  }

  const char *Exec = C.getArgs().MakeArgString(TranslateCmdPath);
  C.addCommand(std::make_unique<Command>(JA, T, ResponseFileSupport::None(),
                                         Exec, CmdArgs, Input, Output));
}

void SPIRV::constructAssembleCommand(Compilation &C, const Tool &T,
                                     const JobAction &JA,
                                     const InputInfo &Output,
                                     const InputInfo &Input,
                                     const llvm::opt::ArgStringList &Args) {
  llvm::opt::ArgStringList CmdArgs(Args);
  CmdArgs.push_back(Input.getFilename());

  assert(Input.getType() == types::TY_PP_Asm && "Unexpected input type");

  CmdArgs.append({"-o", Output.getFilename()});

  // Try to find "spirv-as-<LLVM_VERSION_MAJOR>". Otherwise, fall back to
  // plain "spirv-as".
  using namespace std::string_literals;
  auto VersionedTool = "spirv-as-"s + std::to_string(LLVM_VERSION_MAJOR);
  std::string ExeCand = T.getToolChain().GetProgramPath(VersionedTool.c_str());
  if (!llvm::sys::fs::can_execute(ExeCand))
    ExeCand = T.getToolChain().GetProgramPath("spirv-as");

  if (!llvm::sys::fs::can_execute(ExeCand) &&
      !C.getArgs().hasArg(clang::options::OPT__HASH_HASH_HASH)) {
    C.getDriver().Diag(clang::diag::err_drv_no_spv_tools) << "spirv-as";
    return;
  }
  const char *Exec = C.getArgs().MakeArgString(ExeCand);
  C.addCommand(std::make_unique<Command>(JA, T, ResponseFileSupport::None(),
                                         Exec, CmdArgs, Input, Output));
}

void SPIRV::constructLLVMLinkCommand(Compilation &C, const Tool &T,
                                     const JobAction &JA,
                                     const InputInfo &Output,
                                     const InputInfoList &Inputs,
                                     const llvm::opt::ArgList &Args) {

  ArgStringList LlvmLinkArgs;

  for (auto Input : Inputs)
    LlvmLinkArgs.push_back(Input.getFilename());

  tools::constructLLVMLinkCommand(C, T, JA, Inputs, LlvmLinkArgs, Output, Args);
}

void SPIRV::Translator::ConstructJob(Compilation &C, const JobAction &JA,
                                     const InputInfo &Output,
                                     const InputInfoList &Inputs,
                                     const ArgList &Args,
                                     const char *LinkingOutput) const {
  claimNoWarnArgs(Args);
  if (Inputs.size() != 1)
    llvm_unreachable("Invalid number of input files.");
  constructTranslateCommand(C, *this, JA, Output, Inputs[0], {});
}

void SPIRV::Assembler::ConstructJob(Compilation &C, const JobAction &JA,
                                    const InputInfo &Output,
                                    const InputInfoList &Inputs,
                                    const ArgList &Args,
                                    const char *AssembleOutput) const {
  claimNoWarnArgs(Args);
  if (Inputs.size() != 1)
    llvm_unreachable("Invalid number of input files.");
  constructAssembleCommand(C, *this, JA, Output, Inputs[0], {});
}

clang::driver::Tool *SPIRVToolChain::getAssembler() const {
  if (!Assembler)
    Assembler = std::make_unique<SPIRV::Assembler>(*this);
  return Assembler.get();
}

clang::driver::Tool *SPIRVToolChain::SelectTool(const JobAction &JA) const {
  Action::ActionClass AC = JA.getKind();
  return SPIRVToolChain::getTool(AC);
}

clang::driver::Tool *SPIRVToolChain::getTool(Action::ActionClass AC) const {
  switch (AC) {
  default:
    break;
  case Action::AssembleJobClass:
    return SPIRVToolChain::getAssembler();
  }
  return ToolChain::getTool(AC);
}
clang::driver::Tool *SPIRVToolChain::buildLinker() const {
  return new tools::SPIRV::Linker(*this);
}

// Locates HIP pass plugin for chipstar targets.
static std::string findPassPlugin(const Driver &D,
                                  const llvm::opt::ArgList &Args) {
  llvm::StringRef hipPath = Args.getLastArgValue(options::OPT_hip_path_EQ);
  if (!hipPath.empty()) {
    llvm::SmallString<128> PluginPath(hipPath);
    llvm::sys::path::append(PluginPath, "lib", "libLLVMHipSpvPasses.so");
    if (llvm::sys::fs::exists(PluginPath))
      return PluginPath.str().str();
    PluginPath.assign(hipPath);
    llvm::sys::path::append(PluginPath, "lib", "llvm",
                            "libLLVMHipSpvPasses.so");
    if (llvm::sys::fs::exists(PluginPath))
      return PluginPath.str().str();
  }
  return std::string();
}

void SPIRV::Linker::ConstructJob(Compilation &C, const JobAction &JA,
                                 const InputInfo &Output,
                                 const InputInfoList &Inputs,
                                 const ArgList &Args,
                                 const char *LinkingOutput) const {
  if (JA.getType() == types::TY_LLVM_BC) {
    constructLLVMLinkCommand(C, *this, JA, Output, Inputs, Args);
    return;
  }
  
  const ToolChain &ToolChain = getToolChain();
  auto Triple = ToolChain.getTriple();
  
  // For chipstar targets with new offload driver, implement merge-then-process flow:
  // 1. Merge bitcode with llvm-link
  // 2. Run HipSpvPasses plugin
  // 3. Translate to SPIR-V with llvm-spirv
  // 4. Pass to spirv-link
  if (Triple.getOS() == llvm::Triple::ChipStar) {
    assert(!Inputs.empty() && "Must have at least one input.");
    std::string Name = std::string(llvm::sys::path::stem(Output.getFilename()));
    const char *LinkBCFile = HIP::getTempFile(C, Name + "-link", "bc");
    
    // Step 1: Merge all bitcode files with llvm-link
    ArgStringList LinkArgs;
    for (auto Input : Inputs)
      LinkArgs.push_back(Input.getFilename());
    tools::constructLLVMLinkCommand(C, *this, JA, Inputs, LinkArgs, Output, Args,
                                    LinkBCFile);
    
    // Step 2: Run HipSpvPasses plugin
    const char *ProcessedBCFile = LinkBCFile;
    auto PassPluginPath = findPassPlugin(C.getDriver(), Args);
    if (!PassPluginPath.empty()) {
      const char *PassPathCStr = C.getArgs().MakeArgString(PassPluginPath);
      const char *OptOutput = HIP::getTempFile(C, Name + "-lower", "bc");
      ArgStringList OptArgs{LinkBCFile,     "-load-pass-plugin",
                            PassPathCStr, "-passes=hip-post-link-passes",
                            "-o",         OptOutput};
      // Derive opt path from clang path to ensure we use the same LLVM version
      std::string ClangPath = C.getDriver().getClangProgramPath();
      SmallString<128> OptPath(ClangPath);
      llvm::sys::path::remove_filename(OptPath);
      llvm::sys::path::append(OptPath, "opt");
      const char *Opt = C.getArgs().MakeArgString(OptPath);
      C.addCommand(std::make_unique<Command>(
          JA, *this, ResponseFileSupport::None(), Opt, OptArgs, Inputs, Output));
      ProcessedBCFile = OptOutput;
    }
    
    // Step 3: Translate bitcode to SPIR-V (output goes directly to final output)
    llvm::opt::ArgStringList TrArgs;
    bool HasNoSubArch = Triple.getSubArch() == llvm::Triple::NoSubArch;
    if (HasNoSubArch)
      TrArgs.push_back("--spirv-max-version=1.2");
    TrArgs.push_back("--spirv-ext=-all"
                     ",+SPV_INTEL_function_pointers"
                     ",+SPV_INTEL_subgroups");
    InputInfo TrInput = InputInfo(types::TY_LLVM_BC, ProcessedBCFile, "");
    constructTranslateCommand(C, *this, JA, Output, TrInput, TrArgs);
    return;
  }
  
  // Default flow for non-chipstar targets
  // spirv-link is from SPIRV-Tools (Khronos), not LLVM, so use PATH lookup
  std::string Linker = ToolChain.GetProgramPath(getShortName());
  ArgStringList CmdArgs;
  AddLinkerInputs(getToolChain(), Inputs, Args, CmdArgs, JA);

  CmdArgs.push_back("-o");
  CmdArgs.push_back(Output.getFilename());

  // Use of --sycl-link will call the clang-sycl-linker instead of
  // the default linker (spirv-link).
  if (Args.hasArg(options::OPT_sycl_link))
    Linker = ToolChain.GetProgramPath("clang-sycl-linker");
  else if (!llvm::sys::fs::can_execute(Linker) &&
           !C.getArgs().hasArg(clang::options::OPT__HASH_HASH_HASH)) {
    C.getDriver().Diag(clang::diag::err_drv_no_spv_tools) << getShortName();
    return;
  }
  C.addCommand(std::make_unique<Command>(JA, *this, ResponseFileSupport::None(),
                                         Args.MakeArgString(Linker), CmdArgs,
                                         Inputs, Output));
}

SPIRVToolChain::SPIRVToolChain(const Driver &D, const llvm::Triple &Triple,
                               const ArgList &Args)
    : ToolChain(D, Triple, Args) {
  // TODO: Revisit need/use of --sycl-link option once SYCL toolchain is
  // available and SYCL linking support is moved there.
  NativeLLVMSupport = Args.hasArg(options::OPT_sycl_link);

  // Lookup binaries into the driver directory.
  getProgramPaths().push_back(getDriver().Dir);
}

bool SPIRVToolChain::HasNativeLLVMSupport() const { return NativeLLVMSupport; }
