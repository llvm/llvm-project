//===-- NanoMips.cpp - NaneMips ToolChain Implementations -------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "NanoMips.h"
#include "Arch/Mips.h"
#include "CommonArgs.h"
#include "clang/Driver/Compilation.h"
#include "clang/Driver/Driver.h"
#include "clang/Driver/DriverDiagnostic.h"
#include "clang/Driver/Options.h"
#include "llvm/Option/ArgList.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/VirtualFileSystem.h"

using namespace clang::driver;
using namespace clang::driver::toolchains;
using namespace clang;
using namespace llvm::opt;
using namespace clang::driver::tools;

NanoMips::NanoMips(const Driver &D, const llvm::Triple &Triple,
           const llvm::opt::ArgList &Args) : Generic_ELF(D, Triple, Args) {
  GCCInstallation.init(Triple, Args);
  Multilibs = GCCInstallation.getMultilibs();
  SelectedMultilib = GCCInstallation.getMultilib();

  // The selection of paths to try here is designed to match the patterns which
  // the GCC driver itself uses, as this is part of the GCC-compatible driver.
  // This was determined by running GCC in a fake filesystem, creating all
  // possible permutations of these directories, and seeing which ones it added
  // to the link paths.
  path_list &Paths = getFilePaths();
  std::string SysRoot = computeSysRoot();
  const std::string OSLibDir = "lib";
  const std::string MultiarchTriple = Triple.getTriple();

  Generic_GCC::AddMultilibPaths(D, SysRoot, OSLibDir, MultiarchTriple, Paths);

  addPathIfExists(D, SysRoot + "/lib/" + MultiarchTriple, Paths);
  addPathIfExists(D, SysRoot + "/lib/../" + OSLibDir, Paths);

  // NanoMips multilibs installation has the objects nested under an extra 'lib' dir
  if (GCCInstallation.isValid()) {
    const llvm::Triple &GCCTriple = GCCInstallation.getTriple();
    const std::string &LibPath =
        std::string(GCCInstallation.getParentLibPath());
    addPathIfExists(D,
                    LibPath + "/../" + GCCTriple.str() + "/" + OSLibDir +
                    SelectedMultilib.osSuffix() + "/" + OSLibDir,
                    Paths);
  }

}


void NanoMipsLinker::ConstructJob(Compilation &C, const JobAction &JA,
                                  const InputInfo &Output, const InputInfoList &Inputs,
                                  const llvm::opt::ArgList &Args,
                                  const char *LinkingOutput) const {
  const toolchains::NanoMips &ToolChain =
      static_cast<const toolchains::NanoMips &>(getToolChain());
  const Driver &D = ToolChain.getDriver();

  const bool HasCRTBeginEndFiles =
      ToolChain.getTriple().hasEnvironment() ||
      (ToolChain.getTriple().getVendor() != llvm::Triple::MipsTechnologies);

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

  if (ToolChain.isNoExecStackDefault()) {
    CmdArgs.push_back("-z");
    CmdArgs.push_back("noexecstack");
  }

  if (Args.hasArg(options::OPT_rdynamic))
    CmdArgs.push_back("-export-dynamic");

  if (Args.hasArg(options::OPT_s))
    CmdArgs.push_back("-s");

  ToolChain.addExtraOpts(CmdArgs);

  CmdArgs.push_back("--eh-frame-hdr");

  CmdArgs.push_back("-static");

  CmdArgs.push_back("-o");
  CmdArgs.push_back(Output.getFilename());

  if (!Args.hasArg(options::OPT_nostdlib, options::OPT_nostartfiles)) {
    CmdArgs.push_back(Args.MakeArgString(ToolChain.GetFilePath("crti.o")));

    if (HasCRTBeginEndFiles) {
      std::string P;
      if (ToolChain.GetRuntimeLibType(Args) == ToolChain::RLT_CompilerRT) {
        std::string crtbegin = ToolChain.getCompilerRT(Args, "crtbegin",
                                                       ToolChain::FT_Object);
        if (ToolChain.getVFS().exists(crtbegin))
          P = crtbegin;
      }
      if (P.empty())
        P = ToolChain.GetFilePath("crtbegin.o");
      CmdArgs.push_back(Args.MakeArgString(P));
    }

    // Add crtfastmath.o if available and fast math is enabled.
    ToolChain.addFastMathRuntimeIfAvailable(Args, CmdArgs);
  }

  if (Args.hasArg(options::OPT_mrelax)) {
    CmdArgs.push_back("--relax");
  }

  Args.AddAllArgs(CmdArgs, options::OPT_L);
  Args.AddAllArgs(CmdArgs, options::OPT_u);

  ToolChain.AddFilePathLibArgs(Args, CmdArgs);

  if (D.isUsingLTO()) {
    assert(!Inputs.empty() && "Must have at least one input.");
    addLTOOptions(ToolChain, Args, CmdArgs, Output, Inputs[0],
                  D.getLTOMode() == LTOK_Thin);

    // No object emitter on NanoMips yet, use external assembler for LTO.
    CmdArgs.push_back(Args.MakeArgString("--plugin-opt=-lto-external-asm="
                                         + (getToolChain()
                                            .GetProgramPath("as"))));
    StringRef CPUName;
    StringRef ABIName;
    mips::getMipsCPUAndABI(Args, getToolChain().getTriple(), CPUName, ABIName);

    CmdArgs.push_back("-plugin-opt=-lto-external-asm-arg=-march");
    std::string Arg = "-plugin-opt=-lto-external-asm-arg=";
    Arg += CPUName.data();
    CmdArgs.push_back(Args.MakeArgString(Arg));
    CmdArgs.push_back("-plugin-opt=-lto-external-asm-arg=-EL");
  }

  if (Args.hasArg(options::OPT_Z_Xlinker__no_demangle))
    CmdArgs.push_back("--no-demangle");

  bool NeedsSanitizerDeps = addSanitizerRuntimes(ToolChain, Args, CmdArgs);
  bool NeedsXRayDeps = addXRayRuntime(ToolChain, Args, CmdArgs);
  addLinkerCompressDebugSectionsOption(ToolChain, Args, CmdArgs);
  AddLinkerInputs(ToolChain, Inputs, Args, CmdArgs, JA);
  // The profile runtime also needs access to system libraries.
  getToolChain().addProfileRTLibs(Args, CmdArgs);

  if (D.CCCIsCXX() &&
      !Args.hasArg(options::OPT_nostdlib, options::OPT_nodefaultlibs)) {
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

  if (!Args.hasArg(options::OPT_nostdlib)) {
    if (!Args.hasArg(options::OPT_nodefaultlibs)) {
      CmdArgs.push_back("--start-group");

      if (NeedsSanitizerDeps)
        linkSanitizerRuntimeDeps(ToolChain, CmdArgs);

      if (NeedsXRayDeps)
        linkXRayRuntimeDeps(ToolChain, CmdArgs);

      bool WantPthread = Args.hasArg(options::OPT_pthread) ||
                         Args.hasArg(options::OPT_pthreads);

      // Use the static OpenMP runtime with -static-openmp
      bool StaticOpenMP = Args.hasArg(options::OPT_static_openmp) &&
                          !Args.hasArg(options::OPT_static);

      // FIXME: Only pass GompNeedsRT = true for platforms with libgomp that
      // require librt. Most modern Linux platforms do, but some may not.
      if (addOpenMPRuntime(CmdArgs, ToolChain, Args, StaticOpenMP,
                           JA.isHostOffloading(Action::OFK_OpenMP),
                           /* GompNeedsRT= */ true))
        // OpenMP runtimes implies pthreads when using the GNU toolchain.
        // FIXME: Does this really make sense for all GNU toolchains?
        WantPthread = true;

      AddRunTimeLibs(ToolChain, D, CmdArgs, Args);

      if (WantPthread)
        CmdArgs.push_back("-lpthread");

      if (Args.hasArg(options::OPT_fsplit_stack))
        CmdArgs.push_back("--wrap=pthread_create");

      if (!Args.hasArg(options::OPT_nolibc))
        CmdArgs.push_back("-lc");

      CmdArgs.push_back("--end-group");
    }

    if (!Args.hasArg(options::OPT_nostartfiles)) {
      if (HasCRTBeginEndFiles) {
        std::string P;
        if (ToolChain.GetRuntimeLibType(Args) == ToolChain::RLT_CompilerRT) {
          std::string crtend = ToolChain.getCompilerRT(Args, "crtend",
                                                       ToolChain::FT_Object);
          if (ToolChain.getVFS().exists(crtend))
            P = crtend;
        }
        if (P.empty())
          P = ToolChain.GetFilePath("crtend.o");
        CmdArgs.push_back(Args.MakeArgString(P));
      }
      CmdArgs.push_back(Args.MakeArgString(ToolChain.GetFilePath("crtn.o")));
    }
  }

  Args.AddAllArgs(CmdArgs, options::OPT_T);

  const char *Exec = Args.MakeArgString(ToolChain.GetLinkerPath());
  C.addCommand(std::make_unique<Command>(JA, *this,
                                         ResponseFileSupport::AtFileCurCP(),
                                         Exec, CmdArgs, Inputs, Output));
}

void NanoMips::AddClangSystemIncludeArgs(const llvm::opt::ArgList &DriverArgs,
                                         llvm::opt::ArgStringList &CC1Args) const
{
  if (DriverArgs.hasArg(options::OPT_nostdinc))
    return;

  AddMultilibIncludeArgs(DriverArgs, CC1Args);

  // Add the obvious include dir from the GCC install.
  if (GCCInstallation.isValid()) {
    // Standard libs includes in, eg. <install>/nanomips-elf/include
    llvm::Triple Triple = GCCInstallation.getTriple();
    std::string Install = getDriver().getInstalledDir();
    addSystemInclude(DriverArgs, CC1Args, Install + "/../" + Triple.str() + "/include");

    // GCC's 'fixed' includes directory
    addSystemInclude(DriverArgs, CC1Args, (GCCInstallation.getInstallPath()
                                           + "/include-fixed"));

    // GCC's includes dir for multilibs
    addSystemInclude(DriverArgs, CC1Args, GCCInstallation.getInstallPath() + "/include");
  }

}
