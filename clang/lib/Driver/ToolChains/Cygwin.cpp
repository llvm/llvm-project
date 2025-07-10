//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Cygwin.h"
#include "clang/Config/config.h"
#include "clang/Driver/CommonArgs.h"
#include "clang/Driver/Compilation.h"
#include "clang/Driver/Driver.h"
#include "clang/Driver/Options.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/VirtualFileSystem.h"

using namespace clang::driver;
using namespace clang::driver::toolchains;
using namespace clang;
using namespace llvm::opt;

using tools::addPathIfExists;

Cygwin::Cygwin(const Driver &D, const llvm::Triple &Triple, const ArgList &Args)
    : Generic_GCC(D, Triple, Args) {
  GCCInstallation.init(Triple, Args);
  std::string SysRoot = computeSysRoot();
  ToolChain::path_list &PPaths = getProgramPaths();

  Generic_GCC::PushPPaths(PPaths);

  path_list &Paths = getFilePaths();
  if (GCCInstallation.isValid())
    Paths.push_back(GCCInstallation.getInstallPath().str());

  Generic_GCC::AddMultiarchPaths(D, SysRoot, "lib", Paths);

  // Similar to the logic for GCC above, if we are currently running Clang
  // inside of the requested system root, add its parent library path to those
  // searched.
  // FIXME: It's not clear whether we should use the driver's installed
  // directory ('Dir' below) or the ResourceDir.
  if (StringRef(D.Dir).starts_with(SysRoot))
    addPathIfExists(D, D.Dir + "/../lib", Paths);

  addPathIfExists(D, SysRoot + "/lib", Paths);
  addPathIfExists(D, SysRoot + "/usr/lib", Paths);
  addPathIfExists(D, SysRoot + "/usr/lib/w32api", Paths);
}

llvm::ExceptionHandling Cygwin::GetExceptionModel(const ArgList &Args) const {
  if (getArch() == llvm::Triple::x86_64 || getArch() == llvm::Triple::aarch64 ||
      getArch() == llvm::Triple::arm || getArch() == llvm::Triple::thumb)
    return llvm::ExceptionHandling::WinEH;
  return llvm::ExceptionHandling::DwarfCFI;
}

void Cygwin::AddClangSystemIncludeArgs(const ArgList &DriverArgs,
                                       ArgStringList &CC1Args) const {
  const Driver &D = getDriver();
  std::string SysRoot = computeSysRoot();

  if (DriverArgs.hasArg(clang::driver::options::OPT_nostdinc))
    return;

  if (!DriverArgs.hasArg(options::OPT_nostdlibinc))
    addSystemInclude(DriverArgs, CC1Args, SysRoot + "/usr/local/include");

  if (!DriverArgs.hasArg(options::OPT_nobuiltininc)) {
    SmallString<128> P(D.ResourceDir);
    llvm::sys::path::append(P, "include");
    addSystemInclude(DriverArgs, CC1Args, P);
  }

  if (DriverArgs.hasArg(options::OPT_nostdlibinc))
    return;

  // Check for configure-time C include directories.
  StringRef CIncludeDirs(C_INCLUDE_DIRS);
  if (CIncludeDirs != "") {
    SmallVector<StringRef, 5> Dirs;
    CIncludeDirs.split(Dirs, ":");
    for (StringRef Dir : Dirs) {
      StringRef Prefix =
          llvm::sys::path::is_absolute(Dir) ? "" : StringRef(SysRoot);
      addExternCSystemInclude(DriverArgs, CC1Args, Prefix + Dir);
    }
    return;
  }

  // Lacking those, try to detect the correct set of system includes for the
  // target triple.

  AddMultilibIncludeArgs(DriverArgs, CC1Args);

  // On systems using multiarch, add /usr/include/$triple before
  // /usr/include.
  std::string MultiarchIncludeDir = getTriple().str();
  if (!MultiarchIncludeDir.empty() &&
      D.getVFS().exists(SysRoot + "/usr/include/" + MultiarchIncludeDir))
    addExternCSystemInclude(DriverArgs, CC1Args,
                            SysRoot + "/usr/include/" + MultiarchIncludeDir);

  // Add an include of '/include' directly. This isn't provided by default by
  // system GCCs, but is often used with cross-compiling GCCs, and harmless to
  // add even when Clang is acting as-if it were a system compiler.
  addExternCSystemInclude(DriverArgs, CC1Args, SysRoot + "/include");

  addExternCSystemInclude(DriverArgs, CC1Args, SysRoot + "/usr/include");
  addExternCSystemInclude(DriverArgs, CC1Args, SysRoot + "/usr/include/w32api");
}

static bool getStaticPIE(const ArgList &Args, const ToolChain &TC) {
  bool HasStaticPIE = Args.hasArg(options::OPT_static_pie);
  if (HasStaticPIE && Args.hasArg(options::OPT_no_pie)) {
    const Driver &D = TC.getDriver();
    const llvm::opt::OptTable &Opts = D.getOpts();
    StringRef StaticPIEName = Opts.getOptionName(options::OPT_static_pie);
    StringRef NoPIEName = Opts.getOptionName(options::OPT_nopie);
    D.Diag(diag::err_drv_cannot_mix_options) << StaticPIEName << NoPIEName;
  }
  return HasStaticPIE;
}

static bool getStatic(const ArgList &Args) {
  return Args.hasArg(options::OPT_static) &&
         !Args.hasArg(options::OPT_static_pie);
}

void cygwin::Linker::ConstructJob(Compilation &C, const JobAction &JA,
                                  const InputInfo &Output,
                                  const InputInfoList &Inputs,
                                  const ArgList &Args,
                                  const char *LinkingOutput) const {
  const auto &ToolChain = getToolChain();
  const Driver &D = ToolChain.getDriver();

  const bool IsIAMCU = ToolChain.getTriple().isOSIAMCU();
  const bool IsVE = ToolChain.getTriple().isVE();
  const bool IsStaticPIE = getStaticPIE(Args, ToolChain);
  const bool IsStatic = getStatic(Args);

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

  if (Args.hasArg(options::OPT_s))
    CmdArgs.push_back("-s");

  CmdArgs.push_back("-m");
  switch (ToolChain.getArch()) {
  case llvm::Triple::x86:
    CmdArgs.push_back("i386pe");
    break;
  case llvm::Triple::x86_64:
    CmdArgs.push_back("i386pep");
    break;
  case llvm::Triple::arm:
  case llvm::Triple::thumb:
    // FIXME: this is incorrect for WinCE
    CmdArgs.push_back("thumb2pe");
    break;
  case llvm::Triple::aarch64:
    if (ToolChain.getEffectiveTriple().isWindowsArm64EC())
      CmdArgs.push_back("arm64ecpe");
    else
      CmdArgs.push_back("arm64pe");
    break;
  default:
    D.Diag(diag::err_target_unknown_triple)
        << ToolChain.getEffectiveTriple().str();
  }

  CmdArgs.push_back("--wrap=_Znwm");
  CmdArgs.push_back("--wrap=_Znam");
  CmdArgs.push_back("--wrap=_ZdlPv");
  CmdArgs.push_back("--wrap=_ZdaPv");
  CmdArgs.push_back("--wrap=_ZnwmRKSt9nothrow_t");
  CmdArgs.push_back("--wrap=_ZnamRKSt9nothrow_t");
  CmdArgs.push_back("--wrap=_ZdlPvRKSt9nothrow_t");
  CmdArgs.push_back("--wrap=_ZdaPvRKSt9nothrow_t");

  const bool IsShared = Args.hasArg(options::OPT_shared);
  if (IsShared)
    CmdArgs.push_back("-shared");
  bool IsPIE = false;
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
      IsPIE = Args.hasFlag(options::OPT_pie, options::OPT_no_pie,
                           ToolChain.isPIEDefault(Args));
      if (IsPIE)
        CmdArgs.push_back("-pie");
    }
  }

  CmdArgs.push_back("-o");
  CmdArgs.push_back(Output.getFilename());

  if (!Args.hasArg(options::OPT_nostdlib, options::OPT_nostartfiles,
                   options::OPT_r)) {
    if (IsVE) {
      CmdArgs.push_back("-z");
      CmdArgs.push_back("max-page-size=0x4000000");
    }

    if (IsShared) {
      CmdArgs.push_back("-e");
      CmdArgs.push_back(ToolChain.getTriple().isArch32Bit()
                            ? "__cygwin_dll_entry@12"
                            : "_cygwin_dll_entry");
      CmdArgs.push_back("--enable-auto-image-base");
    }

    if (!IsShared)
      CmdArgs.push_back(Args.MakeArgString(ToolChain.GetFilePath("crt0.o")));
    if (ToolChain.GetRuntimeLibType(Args) == ToolChain::RLT_CompilerRT) {
      std::string crtbegin =
          ToolChain.getCompilerRT(Args, "crtbegin", ToolChain::FT_Object);
      if (ToolChain.getVFS().exists(crtbegin)) {
        std::string P;
        P = crtbegin;
        CmdArgs.push_back(Args.MakeArgString(P));
      }
    }
    if (IsShared)
      CmdArgs.push_back(
          Args.MakeArgString(ToolChain.GetFilePath("crtbeginS.o")));
    else
      CmdArgs.push_back(
          Args.MakeArgString(ToolChain.GetFilePath("crtbegin.o")));

    // Add crtfastmath.o if available and fast math is enabled.
    ToolChain.addFastMathRuntimeIfAvailable(Args, CmdArgs);
  }

  Args.addAllArgs(CmdArgs, {options::OPT_L, options::OPT_u});

  ToolChain.AddFilePathLibArgs(Args, CmdArgs);

  if (D.isUsingLTO())
    tools::addLTOOptions(ToolChain, Args, CmdArgs, Output, Inputs,
                         D.getLTOMode() == LTOK_Thin);

  if (Args.hasArg(options::OPT_Z_Xlinker__no_demangle))
    CmdArgs.push_back("--no-demangle");

  bool NeedsSanitizerDeps =
      tools::addSanitizerRuntimes(ToolChain, Args, CmdArgs);
  bool NeedsXRayDeps = tools::addXRayRuntime(ToolChain, Args, CmdArgs);
  tools::addLinkerCompressDebugSectionsOption(ToolChain, Args, CmdArgs);
  tools::AddLinkerInputs(ToolChain, Inputs, Args, CmdArgs, JA);

  tools::addHIPRuntimeLibArgs(ToolChain, C, Args, CmdArgs);

  // The profile runtime also needs access to system libraries.
  getToolChain().addProfileRTLibs(Args, CmdArgs);

  if (D.CCCIsCXX() &&
      !Args.hasArg(options::OPT_nostdlib, options::OPT_nodefaultlibs,
                   options::OPT_r)) {
    if (ToolChain.ShouldLinkCXXStdlib(Args)) {
      bool OnlyLibstdcxxStatic = Args.hasArg(options::OPT_static_libstdcxx) &&
                                 !Args.hasArg(options::OPT_static);
      if (OnlyLibstdcxxStatic)
        CmdArgs.push_back("-Bstatic");
      ToolChain.AddCXXStdlibLibArgs(Args, CmdArgs);
      if (OnlyLibstdcxxStatic)
        CmdArgs.push_back("-Bdynamic");
    }
    CmdArgs.push_back("-lm");
  }

  // Silence warnings when linking C code with a C++ '-stdlib' argument.
  Args.ClaimAllArgs(options::OPT_stdlib_EQ);

  // Additional linker set-up and flags for Fortran. This is required in order
  // to generate executables. As Fortran runtime depends on the C runtime,
  // these dependencies need to be listed before the C runtime below (i.e.
  // AddRunTimeLibs).
  if (D.IsFlangMode() &&
      !Args.hasArg(options::OPT_nostdlib, options::OPT_nodefaultlibs)) {
    ToolChain.addFortranRuntimeLibraryPath(Args, CmdArgs);
    ToolChain.addFortranRuntimeLibs(Args, CmdArgs);
    CmdArgs.push_back("-lm");
  }

  if (!Args.hasArg(options::OPT_nostdlib, options::OPT_r)) {
    if (!Args.hasArg(options::OPT_nodefaultlibs)) {
      if (IsStatic || IsStaticPIE)
        CmdArgs.push_back("--start-group");

      if (NeedsSanitizerDeps)
        tools::linkSanitizerRuntimeDeps(ToolChain, Args, CmdArgs);

      if (NeedsXRayDeps)
        tools::linkXRayRuntimeDeps(ToolChain, Args, CmdArgs);

      bool WantPthread = Args.hasArg(options::OPT_pthread) ||
                         Args.hasArg(options::OPT_pthreads);

      // Use the static OpenMP runtime with -static-openmp
      bool StaticOpenMP = Args.hasArg(options::OPT_static_openmp) &&
                          !Args.hasArg(options::OPT_static);

      // FIXME: Only pass GompNeedsRT = true for platforms with libgomp that
      // require librt. Most modern Linux platforms do, but some may not.
      if (tools::addOpenMPRuntime(C, CmdArgs, ToolChain, Args, StaticOpenMP,
                                  JA.isHostOffloading(Action::OFK_OpenMP),
                                  /* GompNeedsRT= */ true))
        // OpenMP runtimes implies pthreads when using the GNU toolchain.
        // FIXME: Does this really make sense for all GNU toolchains?
        WantPthread = true;

      tools::AddRunTimeLibs(ToolChain, D, CmdArgs, Args);

      if (WantPthread)
        CmdArgs.push_back("-lpthread");

      if (Args.hasArg(options::OPT_fsplit_stack))
        CmdArgs.push_back("--wrap=pthread_create");

      if (!Args.hasArg(options::OPT_nolibc))
        CmdArgs.push_back("-lc");

      // Cygwin specific
      CmdArgs.push_back("-lcygwin");
      CmdArgs.push_back("-ladvapi32");
      CmdArgs.push_back("-lshell32");
      CmdArgs.push_back("-luser32");
      CmdArgs.push_back("-lkernel32");

      // Add IAMCU specific libs, if needed.
      if (IsIAMCU)
        CmdArgs.push_back("-lgloss");

      if (IsStatic || IsStaticPIE)
        CmdArgs.push_back("--end-group");
      else
        tools::AddRunTimeLibs(ToolChain, D, CmdArgs, Args);

      // Add IAMCU specific libs (outside the group), if needed.
      if (IsIAMCU) {
        CmdArgs.push_back("--as-needed");
        CmdArgs.push_back("-lsoftfp");
        CmdArgs.push_back("--no-as-needed");
      }
    }

    if (!Args.hasArg(options::OPT_nostartfiles) && !IsIAMCU) {
      if (ToolChain.GetRuntimeLibType(Args) == ToolChain::RLT_CompilerRT) {
        std::string crtend =
            ToolChain.getCompilerRT(Args, "crtend", ToolChain::FT_Object);
        if (ToolChain.getVFS().exists(crtend)) {
          std::string P;
          P = crtend;
          CmdArgs.push_back(Args.MakeArgString(P));
        }
      }
      CmdArgs.push_back(Args.MakeArgString(ToolChain.GetFilePath("crtend.o")));
    }
  }

  Args.addAllArgs(CmdArgs, {options::OPT_T, options::OPT_t});

  const char *Exec = Args.MakeArgString(ToolChain.GetLinkerPath());
  C.addCommand(std::make_unique<Command>(JA, *this,
                                         ResponseFileSupport::AtFileCurCP(),
                                         Exec, CmdArgs, Inputs, Output));
}

auto Cygwin::buildLinker() const -> Tool * { return new cygwin::Linker(*this); }
