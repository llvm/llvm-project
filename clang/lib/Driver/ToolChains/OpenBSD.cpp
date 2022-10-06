//===--- OpenBSD.cpp - OpenBSD ToolChain Implementations --------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "OpenBSD.h"
#include "Arch/ARM.h"
#include "Arch/Mips.h"
#include "Arch/Sparc.h"
#include "CommonArgs.h"
#include "clang/Config/config.h"
#include "clang/Driver/Compilation.h"
#include "clang/Driver/Options.h"
#include "clang/Driver/SanitizerArgs.h"
#include "llvm/Option/ArgList.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/VirtualFileSystem.h"

using namespace clang::driver;
using namespace clang::driver::tools;
using namespace clang::driver::toolchains;
using namespace clang;
using namespace llvm::opt;

void openbsd::Assembler::ConstructJob(Compilation &C, const JobAction &JA,
                                      const InputInfo &Output,
                                      const InputInfoList &Inputs,
                                      const ArgList &Args,
                                      const char *LinkingOutput) const {
  const toolchains::OpenBSD &ToolChain =
      static_cast<const toolchains::OpenBSD &>(getToolChain());
  const Driver &D = ToolChain.getDriver();
  const llvm::Triple &Triple = ToolChain.getTriple();

  claimNoWarnArgs(Args);
  ArgStringList CmdArgs;

  switch (ToolChain.getArch()) {
  case llvm::Triple::x86:
    // When building 32-bit code on OpenBSD/amd64, we have to explicitly
    // instruct as in the base system to assemble 32-bit code.
    CmdArgs.push_back("--32");
    break;

  case llvm::Triple::arm:
  case llvm::Triple::armeb: {
    StringRef MArch, MCPU;
    arm::getARMArchCPUFromArgs(Args, MArch, MCPU, /*FromAs*/ true);
    std::string Arch = arm::getARMTargetCPU(MCPU, MArch, Triple);
    CmdArgs.push_back(Args.MakeArgString("-mcpu=" + Arch));
    break;
  }

  case llvm::Triple::ppc:
    CmdArgs.push_back("-mppc");
    CmdArgs.push_back("-many");
    break;

  case llvm::Triple::sparcv9: {
    CmdArgs.push_back("-64");
    std::string CPU = getCPUName(D, Args, Triple);
    CmdArgs.push_back(sparc::getSparcAsmModeForCPU(CPU, Triple));
    AddAssemblerKPIC(ToolChain, Args, CmdArgs);
    break;
  }

  case llvm::Triple::mips64:
  case llvm::Triple::mips64el: {
    StringRef CPUName;
    StringRef ABIName;
    mips::getMipsCPUAndABI(Args, Triple, CPUName, ABIName);

    CmdArgs.push_back("-march");
    CmdArgs.push_back(CPUName.data());

    CmdArgs.push_back("-mabi");
    CmdArgs.push_back(mips::getGnuCompatibleMipsABIName(ABIName).data());

    if (Triple.isLittleEndian())
      CmdArgs.push_back("-EL");
    else
      CmdArgs.push_back("-EB");

    AddAssemblerKPIC(ToolChain, Args, CmdArgs);
    break;
  }

  default:
    break;
  }

  Args.AddAllArgValues(CmdArgs, options::OPT_Wa_COMMA, options::OPT_Xassembler);

  CmdArgs.push_back("-o");
  CmdArgs.push_back(Output.getFilename());

  for (const auto &II : Inputs)
    CmdArgs.push_back(II.getFilename());

  const char *Exec = Args.MakeArgString(ToolChain.GetProgramPath("as"));
  C.addCommand(std::make_unique<Command>(JA, *this,
                                         ResponseFileSupport::AtFileCurCP(),
                                         Exec, CmdArgs, Inputs, Output));
}

void openbsd::Linker::ConstructJob(Compilation &C, const JobAction &JA,
                                   const InputInfo &Output,
                                   const InputInfoList &Inputs,
                                   const ArgList &Args,
                                   const char *LinkingOutput) const {
  const toolchains::OpenBSD &ToolChain =
      static_cast<const toolchains::OpenBSD &>(getToolChain());
  const Driver &D = ToolChain.getDriver();
  const llvm::Triple::ArchType Arch = ToolChain.getArch();
  ArgStringList CmdArgs;

  // Silence warning for "clang -g foo.o -o foo"
  Args.ClaimAllArgs(options::OPT_g_Group);
  // and "clang -emit-llvm foo.o -o foo"
  Args.ClaimAllArgs(options::OPT_emit_llvm);
  // and for "clang -w foo.o -o foo". Other warning options are already
  // handled somewhere else.
  Args.ClaimAllArgs(options::OPT_w);

  if (!D.SysRoot.empty())
    CmdArgs.push_back(Args.MakeArgString("--sysroot=" + D.SysRoot));

  if (Arch == llvm::Triple::mips64)
    CmdArgs.push_back("-EB");
  else if (Arch == llvm::Triple::mips64el)
    CmdArgs.push_back("-EL");

  if (!Args.hasArg(options::OPT_nostdlib, options::OPT_shared)) {
    CmdArgs.push_back("-e");
    CmdArgs.push_back("__start");
  }

  CmdArgs.push_back("--eh-frame-hdr");
  if (Args.hasArg(options::OPT_static)) {
    CmdArgs.push_back("-Bstatic");
  } else {
    if (Args.hasArg(options::OPT_rdynamic))
      CmdArgs.push_back("-export-dynamic");
    CmdArgs.push_back("-Bdynamic");
    if (Args.hasArg(options::OPT_shared)) {
      CmdArgs.push_back("-shared");
    } else if (!Args.hasArg(options::OPT_r)) {
      CmdArgs.push_back("-dynamic-linker");
      CmdArgs.push_back("/usr/libexec/ld.so");
    }
  }

  if (Args.hasArg(options::OPT_pie))
    CmdArgs.push_back("-pie");
  if (Args.hasArg(options::OPT_nopie) || Args.hasArg(options::OPT_pg))
    CmdArgs.push_back("-nopie");

  if (Arch == llvm::Triple::riscv64)
    CmdArgs.push_back("-X");

  if (Output.isFilename()) {
    CmdArgs.push_back("-o");
    CmdArgs.push_back(Output.getFilename());
  } else {
    assert(Output.isNothing() && "Invalid output.");
  }

  if (!Args.hasArg(options::OPT_nostdlib, options::OPT_nostartfiles,
                   options::OPT_r)) {
    const char *crt0 = nullptr;
    const char *crtbegin = nullptr;
    if (!Args.hasArg(options::OPT_shared)) {
      if (Args.hasArg(options::OPT_pg))
        crt0 = "gcrt0.o";
      else if (Args.hasArg(options::OPT_static) &&
               !Args.hasArg(options::OPT_nopie))
        crt0 = "rcrt0.o";
      else
        crt0 = "crt0.o";
      crtbegin = "crtbegin.o";
    } else {
      crtbegin = "crtbeginS.o";
    }

    if (crt0)
      CmdArgs.push_back(Args.MakeArgString(ToolChain.GetFilePath(crt0)));
    CmdArgs.push_back(Args.MakeArgString(ToolChain.GetFilePath(crtbegin)));
  }

  Args.AddAllArgs(CmdArgs, options::OPT_L);
  ToolChain.AddFilePathLibArgs(Args, CmdArgs);
  Args.AddAllArgs(CmdArgs, {options::OPT_T_Group, options::OPT_e,
                            options::OPT_s, options::OPT_t,
                            options::OPT_Z_Flag, options::OPT_r});

  bool NeedsSanitizerDeps = addSanitizerRuntimes(ToolChain, Args, CmdArgs);
  bool NeedsXRayDeps = addXRayRuntime(ToolChain, Args, CmdArgs);
  AddLinkerInputs(ToolChain, Inputs, Args, CmdArgs, JA);

  if (!Args.hasArg(options::OPT_nostdlib, options::OPT_nodefaultlibs,
                   options::OPT_r)) {
    // Use the static OpenMP runtime with -static-openmp
    bool StaticOpenMP = Args.hasArg(options::OPT_static_openmp) &&
                        !Args.hasArg(options::OPT_static);
    addOpenMPRuntime(CmdArgs, ToolChain, Args, StaticOpenMP);

    if (D.CCCIsCXX()) {
      if (ToolChain.ShouldLinkCXXStdlib(Args))
        ToolChain.AddCXXStdlibLibArgs(Args, CmdArgs);
      if (Args.hasArg(options::OPT_pg))
        CmdArgs.push_back("-lm_p");
      else
        CmdArgs.push_back("-lm");
    }
    if (NeedsSanitizerDeps) {
      CmdArgs.push_back(ToolChain.getCompilerRTArgString(Args, "builtins"));
      linkSanitizerRuntimeDeps(ToolChain, CmdArgs);
    }
    if (NeedsXRayDeps) {
      CmdArgs.push_back(ToolChain.getCompilerRTArgString(Args, "builtins"));
      linkXRayRuntimeDeps(ToolChain, CmdArgs);
    }
    // FIXME: For some reason GCC passes -lgcc before adding
    // the default system libraries. Just mimic this for now.
    CmdArgs.push_back("-lcompiler_rt");

    if (Args.hasArg(options::OPT_pthread)) {
      if (!Args.hasArg(options::OPT_shared) && Args.hasArg(options::OPT_pg))
        CmdArgs.push_back("-lpthread_p");
      else
        CmdArgs.push_back("-lpthread");
    }

    if (!Args.hasArg(options::OPT_shared)) {
      if (Args.hasArg(options::OPT_pg))
        CmdArgs.push_back("-lc_p");
      else
        CmdArgs.push_back("-lc");
    }

    CmdArgs.push_back("-lcompiler_rt");
  }

  if (!Args.hasArg(options::OPT_nostdlib, options::OPT_nostartfiles,
                   options::OPT_r)) {
    const char *crtend = nullptr;
    if (!Args.hasArg(options::OPT_shared))
      crtend = "crtend.o";
    else
      crtend = "crtendS.o";

    CmdArgs.push_back(Args.MakeArgString(ToolChain.GetFilePath(crtend)));
  }

  ToolChain.addProfileRTLibs(Args, CmdArgs);

  const char *Exec = Args.MakeArgString(ToolChain.GetLinkerPath());
  C.addCommand(std::make_unique<Command>(JA, *this,
                                         ResponseFileSupport::AtFileCurCP(),
                                         Exec, CmdArgs, Inputs, Output));
}

SanitizerMask OpenBSD::getSupportedSanitizers() const {
  const bool IsX86 = getTriple().getArch() == llvm::Triple::x86;
  const bool IsX86_64 = getTriple().getArch() == llvm::Triple::x86_64;

  // For future use, only UBsan at the moment
  SanitizerMask Res = ToolChain::getSupportedSanitizers();

  if (IsX86 || IsX86_64) {
    Res |= SanitizerKind::Vptr;
    Res |= SanitizerKind::Fuzzer;
    Res |= SanitizerKind::FuzzerNoLink;
  }

  return Res;
}

/// OpenBSD - OpenBSD tool chain which can call as(1) and ld(1) directly.

OpenBSD::OpenBSD(const Driver &D, const llvm::Triple &Triple,
                 const ArgList &Args)
    : Generic_ELF(D, Triple, Args) {
  getFilePaths().push_back(concat(getDriver().SysRoot, "/usr/lib"));
}

void OpenBSD::AddClangSystemIncludeArgs(
    const llvm::opt::ArgList &DriverArgs,
    llvm::opt::ArgStringList &CC1Args) const {
  const Driver &D = getDriver();

  if (DriverArgs.hasArg(clang::driver::options::OPT_nostdinc))
    return;

  if (!DriverArgs.hasArg(options::OPT_nobuiltininc)) {
    SmallString<128> Dir(D.ResourceDir);
    llvm::sys::path::append(Dir, "include");
    addSystemInclude(DriverArgs, CC1Args, Dir.str());
  }

  if (DriverArgs.hasArg(options::OPT_nostdlibinc))
    return;

  // Check for configure-time C include directories.
  StringRef CIncludeDirs(C_INCLUDE_DIRS);
  if (CIncludeDirs != "") {
    SmallVector<StringRef, 5> dirs;
    CIncludeDirs.split(dirs, ":");
    for (StringRef dir : dirs) {
      StringRef Prefix =
          llvm::sys::path::is_absolute(dir) ? StringRef(D.SysRoot) : "";
      addExternCSystemInclude(DriverArgs, CC1Args, Prefix + dir);
    }
    return;
  }

  addExternCSystemInclude(DriverArgs, CC1Args,
                          concat(D.SysRoot, "/usr/include"));
}

void OpenBSD::addLibCxxIncludePaths(const llvm::opt::ArgList &DriverArgs,
                                    llvm::opt::ArgStringList &CC1Args) const {
  addSystemInclude(DriverArgs, CC1Args,
                   concat(getDriver().SysRoot, "/usr/include/c++/v1"));
}

void OpenBSD::AddCXXStdlibLibArgs(const ArgList &Args,
                                  ArgStringList &CmdArgs) const {
  bool Profiling = Args.hasArg(options::OPT_pg);

  CmdArgs.push_back(Profiling ? "-lc++_p" : "-lc++");
  if (Args.hasArg(options::OPT_fexperimental_library))
    CmdArgs.push_back("-lc++experimental");
  CmdArgs.push_back(Profiling ? "-lc++abi_p" : "-lc++abi");
  CmdArgs.push_back(Profiling ? "-lpthread_p" : "-lpthread");
}

std::string OpenBSD::getCompilerRT(const ArgList &Args, StringRef Component,
                                   FileType Type) const {
  if (Component == "builtins") {
    SmallString<128> Path(getDriver().SysRoot);
    llvm::sys::path::append(Path, "/usr/lib/libcompiler_rt.a");
    return std::string(Path.str());
  }
  SmallString<128> P(getDriver().ResourceDir);
  std::string CRTBasename =
      buildCompilerRTBasename(Args, Component, Type, /*AddArch=*/false);
  llvm::sys::path::append(P, "lib", CRTBasename);
  // Checks if this is the base system case which uses a different location.
  if (getVFS().exists(P))
    return std::string(P.str());
  return ToolChain::getCompilerRT(Args, Component, Type);
}

Tool *OpenBSD::buildAssembler() const {
  return new tools::openbsd::Assembler(*this);
}

Tool *OpenBSD::buildLinker() const { return new tools::openbsd::Linker(*this); }

bool OpenBSD::HasNativeLLVMSupport() const { return true; }

ToolChain::UnwindTableLevel
OpenBSD::getDefaultUnwindTableLevel(const ArgList &Args) const {
  switch (getArch()) {
  case llvm::Triple::arm:
    return UnwindTableLevel::None;
  default:
    return UnwindTableLevel::Asynchronous;
  }
}
