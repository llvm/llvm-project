//===---- Serenity.cpp - SerenityOS ToolChain Implementation ----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Serenity.h"
#include "clang/Config/config.h"
#include "clang/Driver/CommonArgs.h"
#include "clang/Driver/Compilation.h"
#include "clang/Driver/Driver.h"
#include "clang/Driver/SanitizerArgs.h"
#include "clang/Options/Options.h"
#include "llvm/Option/ArgList.h"
#include "llvm/Support/VirtualFileSystem.h"
#include <string>

using namespace clang::driver;
using namespace clang::driver::toolchains;
using namespace clang;
using namespace llvm::opt;

void tools::serenity::Linker::ConstructJob(Compilation &C, const JobAction &JA,
                                           const InputInfo &Output,
                                           const InputInfoList &Inputs,
                                           const ArgList &Args,
                                           const char *LinkingOutput) const {
  const auto &TC = static_cast<Generic_ELF const &>(getToolChain());
  const auto &D = TC.getDriver();
  const bool IsShared = Args.hasArg(options::OPT_shared);
  const bool IsStatic =
      Args.hasArg(options::OPT_static) && !Args.hasArg(options::OPT_static_pie);
  const bool IsStaticPIE = Args.hasArg(options::OPT_static_pie);
  ArgStringList CmdArgs;

  if (!D.SysRoot.empty())
    CmdArgs.push_back(Args.MakeArgString("--sysroot=" + D.SysRoot));

  if (IsShared)
    CmdArgs.push_back("-shared");

  if (IsStaticPIE) {
    CmdArgs.push_back("-static");
    CmdArgs.push_back("-pie");
    CmdArgs.push_back("--no-dynamic-linker");
    CmdArgs.push_back("-z");
    CmdArgs.push_back("text");
  } else if (IsStatic) {
    CmdArgs.push_back("-static");
  } else if (!Args.hasArg(options::OPT_r)) {
    if (Args.hasArg(options::OPT_rdynamic))
      CmdArgs.push_back("-export-dynamic");
    if (!IsShared) {
      Arg *A = Args.getLastArg(options::OPT_pie, options::OPT_no_pie,
                               options::OPT_nopie);
      bool IsPIE =
          A ? A->getOption().matches(options::OPT_pie) : TC.isPIEDefault(Args);
      if (IsPIE)
        CmdArgs.push_back("-pie");
      CmdArgs.push_back("-dynamic-linker");
      CmdArgs.push_back(Args.MakeArgString(TC.getDynamicLinker(Args)));
    }
  }

  CmdArgs.push_back("--eh-frame-hdr");

  assert((Output.isFilename() || Output.isNothing()) && "Invalid output.");
  if (Output.isFilename()) {
    CmdArgs.push_back("-o");
    CmdArgs.push_back(Output.getFilename());
  }

  CmdArgs.push_back("-z");
  CmdArgs.push_back("pack-relative-relocs");

  bool HasNoStdLib = Args.hasArg(options::OPT_nostdlib, options::OPT_r);
  bool HasNoStdLibXX = Args.hasArg(options::OPT_nostdlibxx);
  bool HasNoLibC = Args.hasArg(options::OPT_nolibc);
  bool HasNoStartFiles = Args.hasArg(options::OPT_nostartfiles);
  bool HasNoDefaultLibs = Args.hasArg(options::OPT_nodefaultlibs);

  bool ShouldLinkStartFiles = !HasNoStartFiles && !HasNoStdLib;
  bool ShouldLinkCompilerRuntime = !HasNoDefaultLibs && !HasNoStdLib;
  bool ShouldLinkLibC = !HasNoLibC && !HasNoStdLib && !HasNoDefaultLibs;
  bool ShouldLinkLibCXX =
      D.CCCIsCXX() && !HasNoStdLibXX && !HasNoStdLib && !HasNoDefaultLibs;

  if (ShouldLinkStartFiles) {
    if (!IsShared)
      CmdArgs.push_back(Args.MakeArgString(TC.GetFilePath("crt0.o")));

    std::string crtbegin_path;
    if (TC.GetRuntimeLibType(Args) == ToolChain::RLT_CompilerRT) {
      std::string crtbegin =
          TC.getCompilerRT(Args, "crtbegin", ToolChain::FT_Object);
      if (TC.getVFS().exists(crtbegin))
        crtbegin_path = crtbegin;
    }
    if (crtbegin_path.empty())
      crtbegin_path = TC.GetFilePath("crtbeginS.o");
    CmdArgs.push_back(Args.MakeArgString(crtbegin_path));
  }

  Args.addAllArgs(CmdArgs, {options::OPT_L, options::OPT_u});

  TC.AddFilePathLibArgs(Args, CmdArgs);

  if (D.isUsingLTO())
    addLTOOptions(TC, Args, CmdArgs, Output, Inputs,
                  D.getLTOMode() == LTOK_Thin);

  Args.addAllArgs(CmdArgs, {options::OPT_T_Group, options::OPT_s,
                            options::OPT_t, options::OPT_r});

  addLinkerCompressDebugSectionsOption(TC, Args, CmdArgs);

  AddLinkerInputs(TC, Inputs, Args, CmdArgs, JA);

  if (ShouldLinkCompilerRuntime) {
    AddRunTimeLibs(TC, D, CmdArgs, Args);

    // We supply our own sanitizer runtimes that output errors to the
    // Kernel debug log as well as stderr.
    // FIXME: Properly port clang/gcc sanitizers and use those instead.
    const SanitizerArgs &Sanitize = TC.getSanitizerArgs(Args);
    if (Sanitize.needsUbsanRt())
      CmdArgs.push_back("-lubsan");
  }

  if (ShouldLinkLibCXX) {
    bool OnlyLibstdcxxStatic = Args.hasArg(options::OPT_static_libstdcxx) &&
                               !Args.hasArg(options::OPT_static);
    CmdArgs.push_back("--push-state");
    CmdArgs.push_back("--as-needed");
    if (OnlyLibstdcxxStatic)
      CmdArgs.push_back("-Bstatic");
    TC.AddCXXStdlibLibArgs(Args, CmdArgs);
    if (OnlyLibstdcxxStatic)
      CmdArgs.push_back("-Bdynamic");
    CmdArgs.push_back("--pop-state");
  }

  // Silence warnings when linking C code with a C++ '-stdlib' argument.
  Args.ClaimAllArgs(options::OPT_stdlib_EQ);

  if (ShouldLinkLibC)
    CmdArgs.push_back("-lc");

  if (ShouldLinkStartFiles) {
    std::string crtend_path;
    if (TC.GetRuntimeLibType(Args) == ToolChain::RLT_CompilerRT) {
      std::string crtend =
          TC.getCompilerRT(Args, "crtend", ToolChain::FT_Object);
      if (TC.getVFS().exists(crtend))
        crtend_path = crtend;
    }
    if (crtend_path.empty())
      crtend_path = TC.GetFilePath("crtendS.o");
    CmdArgs.push_back(Args.MakeArgString(crtend_path));
  }

  const char *Exec = Args.MakeArgString(TC.GetLinkerPath());
  C.addCommand(std::make_unique<Command>(JA, *this,
                                         ResponseFileSupport::AtFileCurCP(),
                                         Exec, CmdArgs, Inputs, Output));
}

SanitizerMask Serenity::getSupportedSanitizers() const {
  return ToolChain::getSupportedSanitizers() | SanitizerKind::KernelAddress;
}

Serenity::Serenity(const Driver &D, const llvm::Triple &Triple,
                   const ArgList &Args)
    : Generic_ELF(D, Triple, Args) {
  getFilePaths().push_back(concat(getDriver().SysRoot, "/usr/lib"));
}

Tool *Serenity::buildLinker() const {
  return new tools::serenity::Linker(*this);
}

void Serenity::AddClangSystemIncludeArgs(const ArgList &DriverArgs,
                                         ArgStringList &CC1Args) const {
  const Driver &D = getDriver();

  if (DriverArgs.hasArg(options::OPT_nostdinc))
    return;

  if (!DriverArgs.hasArg(options::OPT_nobuiltininc))
    addSystemInclude(DriverArgs, CC1Args, concat(D.ResourceDir, "/include"));

  if (DriverArgs.hasArg(options::OPT_nostdlibinc))
    return;

  addSystemInclude(DriverArgs, CC1Args, concat(D.SysRoot, "/usr/include"));
}
