//===--- SYCL.cpp - SYCL Tool and ToolChain Implementations -----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "SYCL.h"
#include "CommonArgs.h"
#include "InputInfo.h"
#include "clang/Driver/Compilation.h"
#include "clang/Driver/Driver.h"
#include "clang/Driver/DriverDiagnostic.h"
#include "clang/Driver/Options.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"

using namespace clang::driver;
using namespace clang::driver::toolchains;
using namespace clang::driver::tools;
using namespace clang;
using namespace llvm::opt;

const char *SYCL::Linker::constructLLVMSpirvCommand(Compilation &C,
    const JobAction &JA, const InputInfo &Output, StringRef OutputFilePrefix,
    bool ToBc, const char *InputFileName) const {
  // Construct llvm-spirv command.
  // The output is a bc file or vice versa depending on the -r option usage
  // llvm-spirv -r -o a_kernel.bc a_kernel.spv
  // llvm-spirv -o a_kernel.spv a_kernel.bc
  ArgStringList CmdArgs;
  const char *OutputFileName = nullptr;
  if (ToBc) {
    std::string TmpName =
      C.getDriver().GetTemporaryPath(OutputFilePrefix.str() + "-spirv", "bc");
    OutputFileName = C.addTempFile(C.getArgs().MakeArgString(TmpName));
    CmdArgs.push_back("-r");
    CmdArgs.push_back("-o");
    CmdArgs.push_back(OutputFileName);
  } else {
    CmdArgs.push_back("-spirv-no-deref-attr");
    CmdArgs.push_back("-o");
    CmdArgs.push_back(Output.getFilename());
  }
  CmdArgs.push_back(InputFileName);

  SmallString<128> LLVMSpirvPath(C.getDriver().Dir);
  llvm::sys::path::append(LLVMSpirvPath, "llvm-spirv");
  const char *LLVMSpirv = C.getArgs().MakeArgString(LLVMSpirvPath);
  C.addCommand(llvm::make_unique<Command>(JA, *this, LLVMSpirv, CmdArgs, None));
  return OutputFileName;
}

const char *SYCL::Linker::constructLLVMLinkCommand(
    Compilation &C, const JobAction &JA, const ArgList &Args,
    StringRef SubArchName, StringRef OutputFilePrefix,
    const InputInfoList &InputFiles) const {
  ArgStringList CmdArgs;
  // Add the input bc's created by compile step.
  // When offloading, the input file(s) could be from unbundled partially
  // linked archives.  The unbundled information is a list of files and not
  // an actual object/archive.  Take that list and pass those to the linker
  // instead of the original object.
  if (JA.isDeviceOffloading(Action::OFK_SYCL)) {
    // Go through the Inputs to the link.  When a listfile is encountered, we
    // know it is an unbundled generated list.
    // FIXME - properly add objects from list to be removed when compilation is
    // complete.
    for (const auto &II : InputFiles) {
      if (II.getType() == types::TY_Tempfilelist) {
        // Pass the unbundled list with '@' to be processed.
        std::string FileName(II.getFilename());
        CmdArgs.push_back(C.getArgs().MakeArgString("@" + FileName));
      } else
        CmdArgs.push_back(II.getFilename());
    }
  }
  else
    for (const auto &II : InputFiles)
      CmdArgs.push_back(II.getFilename());

  // Add an intermediate output file.
  CmdArgs.push_back("-o");
  SmallString<128> TmpName(C.getDriver().GetTemporaryPath(
                           OutputFilePrefix.str() + "-linked", "bc"));
  const char *OutputFileName =
      C.addTempFile(C.getArgs().MakeArgString(TmpName));
  CmdArgs.push_back(OutputFileName);
  SmallString<128> ExecPath(C.getDriver().Dir);
  llvm::sys::path::append(ExecPath, "llvm-link");
  const char *Exec = C.getArgs().MakeArgString(ExecPath);
  C.addCommand(llvm::make_unique<Command>(JA, *this, Exec, CmdArgs, None));
  return OutputFileName;
}

void SYCL::Linker::constructLlcCommand(Compilation &C, const JobAction &JA,
    const InputInfo &Output, const char *InputFileName) const {
  // Construct llc command.
  // The output is an object file.
  ArgStringList LlcArgs{"-filetype=obj", "-o",  Output.getFilename(),
                        InputFileName};
  SmallString<128> LlcPath(C.getDriver().Dir);
  llvm::sys::path::append(LlcPath, "llc");
  const char *Llc = C.getArgs().MakeArgString(LlcPath);
  C.addCommand(llvm::make_unique<Command>(JA, *this, Llc, LlcArgs, None));
}

// For SYCL the inputs of the linker job are SPIR-V binaries and output is
// a single SPIR-V binary.  Input can also be bitcode when specified by
// the user.
void SYCL::Linker::ConstructJob(Compilation &C, const JobAction &JA,
                                   const InputInfo &Output,
                                   const InputInfoList &Inputs,
                                   const ArgList &Args,
                                   const char *LinkingOutput) const {

  assert((getToolChain().getTriple().getArch() == llvm::Triple::spir ||
          getToolChain().getTriple().getArch() == llvm::Triple::spir64) &&
         "Unsupported target");

  std::string SubArchName = getToolChain().getTriple().getArchName();

  // Prefix for temporary file name.
  std::string Prefix = llvm::sys::path::stem(SubArchName);

  // We want to use llvm-spirv linker to link spirv binaries before putting
  // them into the fat object.
  // Each command outputs different files.
  InputInfoList SpirvInputs;
  for (const auto &II : Inputs) {
    if (!II.isFilename())
      continue;
    if (Args.hasFlag(options::OPT_fsycl_use_bitcode,
                     options::OPT_fno_sycl_use_bitcode, true) ||
        Args.hasArg(options::OPT_foffload_static_lib_EQ))
      SpirvInputs.push_back(II);
    else {
      const char *LLVMSpirvOutputFile =
        constructLLVMSpirvCommand(C, JA, Output, Prefix, true,
                                  II.getFilename());
      SpirvInputs.push_back(InputInfo(types::TY_LLVM_BC, LLVMSpirvOutputFile,
                                      LLVMSpirvOutputFile));
    }
  }
  const char *LLVMLinkOutputFile =
      constructLLVMLinkCommand(C, JA, Args, SubArchName, Prefix, SpirvInputs);
  constructLLVMSpirvCommand(C, JA, Output, Prefix, false, LLVMLinkOutputFile);
}

SYCLToolChain::SYCLToolChain(const Driver &D, const llvm::Triple &Triple,
                             const ToolChain &HostTC, const ArgList &Args)
    : ToolChain(D, Triple, Args), HostTC(HostTC) {
  // Lookup binaries into the driver directory, this is used to
  // discover the clang-offload-bundler executable.
  getProgramPaths().push_back(getDriver().Dir);
}

void SYCLToolChain::addClangTargetOptions(
    const llvm::opt::ArgList &DriverArgs,
    llvm::opt::ArgStringList &CC1Args,
    Action::OffloadKind DeviceOffloadingKind) const {
  HostTC.addClangTargetOptions(DriverArgs, CC1Args, DeviceOffloadingKind);

  assert(DeviceOffloadingKind == Action::OFK_SYCL &&
         "Only SYCL offloading kinds are supported");
}

llvm::opt::DerivedArgList *
SYCLToolChain::TranslateArgs(const llvm::opt::DerivedArgList &Args,
                             StringRef BoundArch,
                             Action::OffloadKind DeviceOffloadKind) const {
  DerivedArgList *DAL =
      HostTC.TranslateArgs(Args, BoundArch, DeviceOffloadKind);
  if (!DAL)
    DAL = new DerivedArgList(Args.getBaseArgs());

  const OptTable &Opts = getDriver().getOpts();

  for (Arg *A : Args) {
    DAL->append(A);
  }

  if (!BoundArch.empty()) {
    DAL->eraseArg(options::OPT_march_EQ);
    DAL->AddJoinedArg(nullptr, Opts.getOption(options::OPT_march_EQ),
                      BoundArch);
  }
  return DAL;
}

Tool *SYCLToolChain::buildLinker() const {
  assert(getTriple().getArch() == llvm::Triple::spir ||
         getTriple().getArch() == llvm::Triple::spir64);
  return new tools::SYCL::Linker(*this);
}

void SYCLToolChain::addClangWarningOptions(ArgStringList &CC1Args) const {
  HostTC.addClangWarningOptions(CC1Args);
}

ToolChain::CXXStdlibType
SYCLToolChain::GetCXXStdlibType(const ArgList &Args) const {
  return HostTC.GetCXXStdlibType(Args);
}

void SYCLToolChain::AddClangSystemIncludeArgs(const ArgList &DriverArgs,
                                              ArgStringList &CC1Args) const {
  HostTC.AddClangSystemIncludeArgs(DriverArgs, CC1Args);
}

void SYCLToolChain::AddClangCXXStdlibIncludeArgs(const ArgList &Args,
                                                 ArgStringList &CC1Args) const {
  HostTC.AddClangCXXStdlibIncludeArgs(Args, CC1Args);
}

