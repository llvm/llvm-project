//===--- MinGW.cpp - MinGWToolChain Implementation ------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "MinGW.h"
#include "InputInfo.h"
#include "CommonArgs.h"
#include "clang/Driver/Compilation.h"
#include "clang/Driver/Driver.h"
#include "clang/Driver/DriverDiagnostic.h"
#include "clang/Driver/Options.h"
#include "llvm/Option/ArgList.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include <system_error>

using namespace clang::diag;
using namespace clang::driver;
using namespace clang;
using namespace llvm::opt;

/// MinGW Tools
void tools::MinGW::Assembler::ConstructJob(Compilation &C, const JobAction &JA,
                                           const InputInfo &Output,
                                           const InputInfoList &Inputs,
                                           const ArgList &Args,
                                           const char *LinkingOutput) const {
  claimNoWarnArgs(Args);
  ArgStringList CmdArgs;

  if (getToolChain().getArch() == llvm::Triple::x86) {
    CmdArgs.push_back("--32");
  } else if (getToolChain().getArch() == llvm::Triple::x86_64) {
    CmdArgs.push_back("--64");
  }

  Args.AddAllArgValues(CmdArgs, options::OPT_Wa_COMMA, options::OPT_Xassembler);

  CmdArgs.push_back("-o");
  CmdArgs.push_back(Output.getFilename());

  for (const auto &II : Inputs)
    CmdArgs.push_back(II.getFilename());

  const char *Exec = Args.MakeArgString(getToolChain().GetProgramPath("as"));
  C.addCommand(llvm::make_unique<Command>(JA, *this, Exec, CmdArgs, Inputs));

  if (Args.hasArg(options::OPT_gsplit_dwarf))
    SplitDebugInfo(getToolChain(), C, *this, JA, Args, Output,
                   SplitDebugName(Args, Inputs[0]));
}

void tools::MinGW::Linker::AddLibGCC(const ArgList &Args,
                                     ArgStringList &CmdArgs) const {
  if (Args.hasArg(options::OPT_mthreads))
    CmdArgs.push_back("-lmingwthrd");
  CmdArgs.push_back("-lmingw32");

  // Make use of compiler-rt if --rtlib option is used
  ToolChain::RuntimeLibType RLT = getToolChain().GetRuntimeLibType(Args);
  if (RLT == ToolChain::RLT_Libgcc) {
    bool Static = Args.hasArg(options::OPT_static_libgcc) ||
                  Args.hasArg(options::OPT_static);
    bool Shared = Args.hasArg(options::OPT_shared);
    bool CXX = getToolChain().getDriver().CCCIsCXX();

    if (Static || (!CXX && !Shared)) {
      CmdArgs.push_back("-lgcc");
      CmdArgs.push_back("-lgcc_eh");
    } else {
      CmdArgs.push_back("-lgcc_s");
      CmdArgs.push_back("-lgcc");
    }
  } else {
    AddRunTimeLibs(getToolChain(), getToolChain().getDriver(), CmdArgs, Args);
  }

  CmdArgs.push_back("-lmoldname");
  CmdArgs.push_back("-lmingwex");
  CmdArgs.push_back("-lmsvcrt");
}

void tools::MinGW::Linker::ConstructJob(Compilation &C, const JobAction &JA,
                                        const InputInfo &Output,
                                        const InputInfoList &Inputs,
                                        const ArgList &Args,
                                        const char *LinkingOutput) const {
  const ToolChain &TC = getToolChain();
  const Driver &D = TC.getDriver();
  // const SanitizerArgs &Sanitize = TC.getSanitizerArgs();

  ArgStringList CmdArgs;

  // Silence warning for "clang -g foo.o -o foo"
  Args.ClaimAllArgs(options::OPT_g_Group);
  // and "clang -emit-llvm foo.o -o foo"
  Args.ClaimAllArgs(options::OPT_emit_llvm);
  // and for "clang -w foo.o -o foo". Other warning options are already
  // handled somewhere else.
  Args.ClaimAllArgs(options::OPT_w);

  StringRef LinkerName = Args.getLastArgValue(options::OPT_fuse_ld_EQ, "ld");
  if (LinkerName.equals_lower("lld")) {
    CmdArgs.push_back("-flavor");
    CmdArgs.push_back("gnu");
  } else if (!LinkerName.equals_lower("ld")) {
    D.Diag(diag::err_drv_unsupported_linker) << LinkerName;
  }

  if (!D.SysRoot.empty())
    CmdArgs.push_back(Args.MakeArgString("--sysroot=" + D.SysRoot));

  if (Args.hasArg(options::OPT_s))
    CmdArgs.push_back("-s");

  CmdArgs.push_back("-m");
  if (TC.getArch() == llvm::Triple::x86)
    CmdArgs.push_back("i386pe");
  if (TC.getArch() == llvm::Triple::x86_64)
    CmdArgs.push_back("i386pep");
  if (TC.getArch() == llvm::Triple::arm)
    CmdArgs.push_back("thumb2pe");

  if (Args.hasArg(options::OPT_mwindows)) {
    CmdArgs.push_back("--subsystem");
    CmdArgs.push_back("windows");
  } else if (Args.hasArg(options::OPT_mconsole)) {
    CmdArgs.push_back("--subsystem");
    CmdArgs.push_back("console");
  }

  if (Args.hasArg(options::OPT_static))
    CmdArgs.push_back("-Bstatic");
  else {
    if (Args.hasArg(options::OPT_mdll))
      CmdArgs.push_back("--dll");
    else if (Args.hasArg(options::OPT_shared))
      CmdArgs.push_back("--shared");
    CmdArgs.push_back("-Bdynamic");
    if (Args.hasArg(options::OPT_mdll) || Args.hasArg(options::OPT_shared)) {
      CmdArgs.push_back("-e");
      if (TC.getArch() == llvm::Triple::x86)
        CmdArgs.push_back("_DllMainCRTStartup@12");
      else
        CmdArgs.push_back("DllMainCRTStartup");
      CmdArgs.push_back("--enable-auto-image-base");
    }
  }

  CmdArgs.push_back("-o");
  CmdArgs.push_back(Output.getFilename());

  Args.AddAllArgs(CmdArgs, options::OPT_e);
  // FIXME: add -N, -n flags
  Args.AddLastArg(CmdArgs, options::OPT_r);
  Args.AddLastArg(CmdArgs, options::OPT_s);
  Args.AddLastArg(CmdArgs, options::OPT_t);
  Args.AddAllArgs(CmdArgs, options::OPT_u_Group);
  Args.AddLastArg(CmdArgs, options::OPT_Z_Flag);

  if (!Args.hasArg(options::OPT_nostdlib, options::OPT_nostartfiles)) {
    if (Args.hasArg(options::OPT_shared) || Args.hasArg(options::OPT_mdll)) {
      CmdArgs.push_back(Args.MakeArgString(TC.GetFilePath("dllcrt2.o")));
    } else {
      if (Args.hasArg(options::OPT_municode))
        CmdArgs.push_back(Args.MakeArgString(TC.GetFilePath("crt2u.o")));
      else
        CmdArgs.push_back(Args.MakeArgString(TC.GetFilePath("crt2.o")));
    }
    if (Args.hasArg(options::OPT_pg))
      CmdArgs.push_back(Args.MakeArgString(TC.GetFilePath("gcrt2.o")));
    CmdArgs.push_back(Args.MakeArgString(TC.GetFilePath("crtbegin.o")));
  }

  Args.AddAllArgs(CmdArgs, options::OPT_L);
  TC.AddFilePathLibArgs(Args, CmdArgs);
  AddLinkerInputs(TC, Inputs, Args, CmdArgs, JA);

  // TODO: Add ASan stuff here

  // TODO: Add profile stuff here

  if (D.CCCIsCXX() &&
      !Args.hasArg(options::OPT_nostdlib, options::OPT_nodefaultlibs)) {
    bool OnlyLibstdcxxStatic = Args.hasArg(options::OPT_static_libstdcxx) &&
                               !Args.hasArg(options::OPT_static);
    if (OnlyLibstdcxxStatic)
      CmdArgs.push_back("-Bstatic");
    TC.AddCXXStdlibLibArgs(Args, CmdArgs);
    if (OnlyLibstdcxxStatic)
      CmdArgs.push_back("-Bdynamic");
  }

  if (!Args.hasArg(options::OPT_nostdlib)) {
    if (!Args.hasArg(options::OPT_nodefaultlibs)) {
      if (Args.hasArg(options::OPT_static))
        CmdArgs.push_back("--start-group");

      if (Args.hasArg(options::OPT_fstack_protector) ||
          Args.hasArg(options::OPT_fstack_protector_strong) ||
          Args.hasArg(options::OPT_fstack_protector_all)) {
        CmdArgs.push_back("-lssp_nonshared");
        CmdArgs.push_back("-lssp");
      }
      if (Args.hasArg(options::OPT_fopenmp))
        CmdArgs.push_back("-lgomp");

      AddLibGCC(Args, CmdArgs);

      if (Args.hasArg(options::OPT_pg))
        CmdArgs.push_back("-lgmon");

      if (Args.hasArg(options::OPT_pthread))
        CmdArgs.push_back("-lpthread");

      // add system libraries
      if (Args.hasArg(options::OPT_mwindows)) {
        CmdArgs.push_back("-lgdi32");
        CmdArgs.push_back("-lcomdlg32");
      }
      CmdArgs.push_back("-ladvapi32");
      CmdArgs.push_back("-lshell32");
      CmdArgs.push_back("-luser32");
      CmdArgs.push_back("-lkernel32");

      if (Args.hasArg(options::OPT_static))
        CmdArgs.push_back("--end-group");
      else if (!LinkerName.equals_lower("lld"))
        AddLibGCC(Args, CmdArgs);
    }

    if (!Args.hasArg(options::OPT_nostartfiles)) {
      // Add crtfastmath.o if available and fast math is enabled.
      TC.AddFastMathRuntimeIfAvailable(Args, CmdArgs);

      CmdArgs.push_back(Args.MakeArgString(TC.GetFilePath("crtend.o")));
    }
  }
  const char *Exec = Args.MakeArgString(TC.GetProgramPath(LinkerName.data()));
  C.addCommand(llvm::make_unique<Command>(JA, *this, Exec, CmdArgs, Inputs));
}

// Simplified from Generic_GCC::GCCInstallationDetector::ScanLibDirForGCCTriple.
static bool findGccVersion(StringRef LibDir, std::string &GccLibDir,
                           std::string &Ver) {
  auto Version = toolchains::Generic_GCC::GCCVersion::Parse("0.0.0");
  std::error_code EC;
  for (llvm::sys::fs::directory_iterator LI(LibDir, EC), LE; !EC && LI != LE;
       LI = LI.increment(EC)) {
    StringRef VersionText = llvm::sys::path::filename(LI->path());
    auto CandidateVersion =
        toolchains::Generic_GCC::GCCVersion::Parse(VersionText);
    if (CandidateVersion.Major == -1)
      continue;
    if (CandidateVersion <= Version)
      continue;
    Ver = VersionText;
    GccLibDir = LI->path();
  }
  return Ver.size();
}

void toolchains::MinGW::findGccLibDir() {
  llvm::SmallVector<llvm::SmallString<32>, 2> Archs;
  Archs.emplace_back(getTriple().getArchName());
  Archs[0] += "-w64-mingw32";
  Archs.emplace_back("mingw32");
  Arch = Archs[0].str();
  // lib: Arch Linux, Ubuntu, Windows
  // lib64: openSUSE Linux
  for (StringRef CandidateLib : {"lib", "lib64"}) {
    for (StringRef CandidateArch : Archs) {
      llvm::SmallString<1024> LibDir(Base);
      llvm::sys::path::append(LibDir, CandidateLib, "gcc", CandidateArch);
      if (findGccVersion(LibDir, GccLibDir, Ver)) {
        Arch = CandidateArch;
        return;
      }
    }
  }
}

llvm::ErrorOr<std::string> toolchains::MinGW::findGcc() {
  llvm::SmallVector<llvm::SmallString<32>, 2> Gccs;
  Gccs.emplace_back(getTriple().getArchName());
  Gccs[0] += "-w64-mingw32-gcc";
  Gccs.emplace_back("mingw32-gcc");
  // Please do not add "gcc" here
  for (StringRef CandidateGcc : Gccs)
    if (llvm::ErrorOr<std::string> GPPName = llvm::sys::findProgramByName(CandidateGcc))
      return GPPName;
  return make_error_code(std::errc::no_such_file_or_directory);
}

toolchains::MinGW::MinGW(const Driver &D, const llvm::Triple &Triple,
                         const ArgList &Args)
    : ToolChain(D, Triple, Args), CudaInstallation(D, Triple, Args) {
  getProgramPaths().push_back(getDriver().getInstalledDir());

  if (getDriver().SysRoot.size())
    Base = getDriver().SysRoot;
  else if (llvm::ErrorOr<std::string> GPPName = findGcc())
    Base = llvm::sys::path::parent_path(
        llvm::sys::path::parent_path(GPPName.get()));
  else
    Base = llvm::sys::path::parent_path(getDriver().getInstalledDir());

  Base += llvm::sys::path::get_separator();
  findGccLibDir();
  // GccLibDir must precede Base/lib so that the
  // correct crtbegin.o ,cetend.o would be found.
  getFilePaths().push_back(GccLibDir);
  getFilePaths().push_back(
      (Base + Arch + llvm::sys::path::get_separator() + "lib").str());
  getFilePaths().push_back(Base + "lib");
  // openSUSE
  getFilePaths().push_back(Base + Arch + "/sys-root/mingw/lib");
}

bool toolchains::MinGW::IsIntegratedAssemblerDefault() const { return true; }

Tool *toolchains::MinGW::getTool(Action::ActionClass AC) const {
  switch (AC) {
  case Action::PreprocessJobClass:
    if (!Preprocessor)
      Preprocessor.reset(new tools::gcc::Preprocessor(*this));
    return Preprocessor.get();
  case Action::CompileJobClass:
    if (!Compiler)
      Compiler.reset(new tools::gcc::Compiler(*this));
    return Compiler.get();
  default:
    return ToolChain::getTool(AC);
  }
}

Tool *toolchains::MinGW::buildAssembler() const {
  return new tools::MinGW::Assembler(*this);
}

Tool *toolchains::MinGW::buildLinker() const {
  return new tools::MinGW::Linker(*this);
}

bool toolchains::MinGW::IsUnwindTablesDefault() const {
  return getArch() == llvm::Triple::x86_64;
}

bool toolchains::MinGW::isPICDefault() const {
  return getArch() == llvm::Triple::x86_64;
}

bool toolchains::MinGW::isPIEDefault() const { return false; }

bool toolchains::MinGW::isPICDefaultForced() const {
  return getArch() == llvm::Triple::x86_64;
}

bool toolchains::MinGW::UseSEHExceptions() const {
  return getArch() == llvm::Triple::x86_64;
}

void toolchains::MinGW::AddCudaIncludeArgs(const ArgList &DriverArgs,
                                           ArgStringList &CC1Args) const {
  CudaInstallation.AddCudaIncludeArgs(DriverArgs, CC1Args);
}

void toolchains::MinGW::printVerboseInfo(raw_ostream &OS) const {
  CudaInstallation.print(OS);
}

// Include directories for various hosts:

// Windows, mingw.org
// c:\mingw\lib\gcc\mingw32\4.8.1\include\c++
// c:\mingw\lib\gcc\mingw32\4.8.1\include\c++\mingw32
// c:\mingw\lib\gcc\mingw32\4.8.1\include\c++\backward
// c:\mingw\include
// c:\mingw\mingw32\include

// Windows, mingw-w64 mingw-builds
// c:\mingw32\i686-w64-mingw32\include
// c:\mingw32\i686-w64-mingw32\include\c++
// c:\mingw32\i686-w64-mingw32\include\c++\i686-w64-mingw32
// c:\mingw32\i686-w64-mingw32\include\c++\backward

// Windows, mingw-w64 msys2
// c:\msys64\mingw32\include
// c:\msys64\mingw32\i686-w64-mingw32\include
// c:\msys64\mingw32\include\c++\4.9.2
// c:\msys64\mingw32\include\c++\4.9.2\i686-w64-mingw32
// c:\msys64\mingw32\include\c++\4.9.2\backward

// openSUSE
// /usr/lib64/gcc/x86_64-w64-mingw32/5.1.0/include/c++
// /usr/lib64/gcc/x86_64-w64-mingw32/5.1.0/include/c++/x86_64-w64-mingw32
// /usr/lib64/gcc/x86_64-w64-mingw32/5.1.0/include/c++/backward
// /usr/x86_64-w64-mingw32/sys-root/mingw/include

// Arch Linux
// /usr/i686-w64-mingw32/include/c++/5.1.0
// /usr/i686-w64-mingw32/include/c++/5.1.0/i686-w64-mingw32
// /usr/i686-w64-mingw32/include/c++/5.1.0/backward
// /usr/i686-w64-mingw32/include

// Ubuntu
// /usr/include/c++/4.8
// /usr/include/c++/4.8/x86_64-w64-mingw32
// /usr/include/c++/4.8/backward
// /usr/x86_64-w64-mingw32/include

void toolchains::MinGW::AddClangSystemIncludeArgs(const ArgList &DriverArgs,
                                                  ArgStringList &CC1Args) const {
  if (DriverArgs.hasArg(options::OPT_nostdinc))
    return;

  if (!DriverArgs.hasArg(options::OPT_nobuiltininc)) {
    SmallString<1024> P(getDriver().ResourceDir);
    llvm::sys::path::append(P, "include");
    addSystemInclude(DriverArgs, CC1Args, P.str());
  }

  if (DriverArgs.hasArg(options::OPT_nostdlibinc))
    return;

  if (GetRuntimeLibType(DriverArgs) == ToolChain::RLT_Libgcc) {
    // openSUSE
    addSystemInclude(DriverArgs, CC1Args,
                     Base + Arch + "/sys-root/mingw/include");
  }

  addSystemInclude(DriverArgs, CC1Args,
                   Base + Arch + llvm::sys::path::get_separator() + "include");
  addSystemInclude(DriverArgs, CC1Args, Base + "include");
}

void toolchains::MinGW::AddClangCXXStdlibIncludeArgs(
    const ArgList &DriverArgs, ArgStringList &CC1Args) const {
  if (DriverArgs.hasArg(options::OPT_nostdlibinc) ||
      DriverArgs.hasArg(options::OPT_nostdincxx))
    return;

  switch (GetCXXStdlibType(DriverArgs)) {
  case ToolChain::CST_Libcxx:
    addSystemInclude(DriverArgs, CC1Args,
                     Base + "include" + llvm::sys::path::get_separator() +
                         "c++" + llvm::sys::path::get_separator() + "v1");
    break;

  case ToolChain::CST_Libstdcxx:
    llvm::SmallVector<llvm::SmallString<1024>, 4> CppIncludeBases;
    CppIncludeBases.emplace_back(Base);
    llvm::sys::path::append(CppIncludeBases[0], Arch, "include", "c++");
    CppIncludeBases.emplace_back(Base);
    llvm::sys::path::append(CppIncludeBases[1], Arch, "include", "c++", Ver);
    CppIncludeBases.emplace_back(Base);
    llvm::sys::path::append(CppIncludeBases[2], "include", "c++", Ver);
    CppIncludeBases.emplace_back(GccLibDir);
    llvm::sys::path::append(CppIncludeBases[3], "include", "c++");
    for (auto &CppIncludeBase : CppIncludeBases) {
      addSystemInclude(DriverArgs, CC1Args, CppIncludeBase);
      CppIncludeBase += llvm::sys::path::get_separator();
      addSystemInclude(DriverArgs, CC1Args, CppIncludeBase + Arch);
      addSystemInclude(DriverArgs, CC1Args, CppIncludeBase + "backward");
    }
    break;
  }
}
