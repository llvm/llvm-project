//===--- UEFI.cpp - UEFI ToolChain Implementations -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "UEFI.h"
#include "clang/Config/config.h"
#include "clang/Driver/CommonArgs.h"
#include "clang/Driver/Compilation.h"
#include "clang/Driver/Driver.h"
#include "clang/Driver/Options.h"
#include "clang/Driver/SanitizerArgs.h"
#include "llvm/Option/Arg.h"
#include "llvm/Option/ArgList.h"
#include "llvm/Support/VirtualFileSystem.h"
#include "llvm/TargetParser/Host.h"

using namespace clang::driver;
using namespace clang::driver::toolchains;
using namespace clang;
using namespace llvm::opt;

UEFI::UEFI(const Driver &D, const llvm::Triple &Triple, const ArgList &Args)
    : ToolChain(D, Triple, Args) {
  getProgramPaths().push_back(getDriver().Dir);
}

Tool *UEFI::buildLinker() const { return new tools::uefi::Linker(*this); }

void UEFI::AddClangSystemIncludeArgs(const ArgList &DriverArgs,
                                     ArgStringList &CC1Args) const {
  if (DriverArgs.hasArg(options::OPT_nostdinc))
    return;

  if (!DriverArgs.hasArg(options::OPT_nobuiltininc)) {
    SmallString<128> Dir(getDriver().ResourceDir);
    llvm::sys::path::append(Dir, "include");
    addSystemInclude(DriverArgs, CC1Args, Dir.str());
  }

  if (DriverArgs.hasArg(options::OPT_nostdlibinc))
    return;

  if (std::optional<std::string> Path = getStdlibIncludePath())
    addSystemInclude(DriverArgs, CC1Args, *Path);
}

void tools::uefi::Linker::ConstructJob(Compilation &C, const JobAction &JA,
                                       const InputInfo &Output,
                                       const InputInfoList &Inputs,
                                       const ArgList &Args,
                                       const char *LinkingOutput) const {
  ArgStringList CmdArgs;
  auto &TC = static_cast<const toolchains::UEFI &>(getToolChain());

  assert((Output.isFilename() || Output.isNothing()) && "invalid output");
  if (Output.isFilename())
    CmdArgs.push_back(
        Args.MakeArgString(std::string("/out:") + Output.getFilename()));

  CmdArgs.push_back("/nologo");

  // Default entry function name according to the TianoCore reference
  // implementation is EfiMain.  -Wl,/subsystem:... or -Wl,/entry:... can
  // override these since they will be added later in AddLinkerInputs.
  CmdArgs.push_back("/subsystem:efi_application");
  CmdArgs.push_back("/entry:EfiMain");

  // "Terminal Service Aware" flag is not needed for UEFI applications.
  CmdArgs.push_back("/tsaware:no");

  if (Args.hasArg(options::OPT_g_Group, options::OPT__SLASH_Z7))
    CmdArgs.push_back("/debug");

  Args.AddAllArgValues(CmdArgs, options::OPT__SLASH_link);

  AddLinkerInputs(TC, Inputs, Args, CmdArgs, JA);

  // Sample these options first so they are claimed even under -nostdlib et al.
  bool NoLibc = Args.hasArg(options::OPT_nolibc);
  if (!Args.hasArg(options::OPT_nostdlib, options::OPT_nodefaultlibs,
                   options::OPT_r)) {
    addSanitizerRuntimes(TC, Args, CmdArgs);

    addXRayRuntime(TC, Args, CmdArgs);

    TC.addProfileRTLibs(Args, CmdArgs);

    // TODO: When compiler-rt/lib/builtins is ready, enable this call:
    // AddRunTimeLibs(TC, TC.getDriver(), CmdArgs, Args);

    if (!NoLibc) {
      // TODO: When there is a libc ready, add it here.
    }
  }

  // This should ideally be handled by ToolChain::GetLinkerPath but we need
  // to special case some linker paths. In the case of lld, we need to
  // translate 'lld' into 'lld-link'.
  StringRef Linker = Args.getLastArgValue(options::OPT_fuse_ld_EQ,
                                          TC.getDriver().getPreferredLinker());
  if (Linker.empty() || Linker == "lld")
    Linker = "lld-link";

  auto LinkerPath = TC.GetProgramPath(Linker.str().c_str());
  auto LinkCmd = std::make_unique<Command>(
      JA, *this, ResponseFileSupport::AtFileUTF16(),
      Args.MakeArgString(LinkerPath), CmdArgs, Inputs, Output);
  C.addCommand(std::move(LinkCmd));
}
