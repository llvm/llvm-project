//===--- MinGWToolChain.cpp - MinGWToolChain Implementation ---------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "ToolChains.h"
#include "clang/Driver/Driver.h"
#include "clang/Driver/Options.h"
#include "llvm/Option/ArgList.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"

using namespace clang::diag;
using namespace clang::driver;
using namespace clang::driver::toolchains;
using namespace clang;
using namespace llvm::opt;

// Simplified from Generic_GCC::GCCInstallationDetector::ScanLibDirForGCCTriple.
static bool findGccVersion(StringRef LibDir, std::string &GccLibDir,
                           std::string &Ver) {
  Generic_GCC::GCCVersion Version = Generic_GCC::GCCVersion::Parse("0.0.0");
  std::error_code EC;
  for (llvm::sys::fs::directory_iterator LI(LibDir, EC), LE; !EC && LI != LE;
       LI = LI.increment(EC)) {
    StringRef VersionText = llvm::sys::path::filename(LI->path());
    Generic_GCC::GCCVersion CandidateVersion =
        Generic_GCC::GCCVersion::Parse(VersionText);
    if (CandidateVersion.Major == -1)
      continue;
    if (CandidateVersion <= Version)
      continue;
    Ver = VersionText;
    GccLibDir = LI->path();
  }
  return Ver.size();
}

void MinGW::findGccLibDir() {
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

MinGW::MinGW(const Driver &D, const llvm::Triple &Triple, const ArgList &Args)
    : ToolChain(D, Triple, Args), CudaInstallation(D, Triple, Args) {
  getProgramPaths().push_back(getDriver().getInstalledDir());

// In Windows there aren't any standard install locations, we search
// for gcc on the PATH. In Linux the base is always /usr.
#ifdef LLVM_ON_WIN32
  if (getDriver().SysRoot.size())
    Base = getDriver().SysRoot;
  else if (llvm::ErrorOr<std::string> GPPName =
               llvm::sys::findProgramByName("gcc"))
    Base = llvm::sys::path::parent_path(
        llvm::sys::path::parent_path(GPPName.get()));
  else
    Base = llvm::sys::path::parent_path(getDriver().getInstalledDir());
#else
  if (getDriver().SysRoot.size())
    Base = getDriver().SysRoot;
  else
    Base = "/usr";
#endif

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

bool MinGW::IsIntegratedAssemblerDefault() const { return true; }

Tool *MinGW::getTool(Action::ActionClass AC) const {
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

Tool *MinGW::buildAssembler() const {
  return new tools::MinGW::Assembler(*this);
}

Tool *MinGW::buildLinker() const { return new tools::MinGW::Linker(*this); }

bool MinGW::IsUnwindTablesDefault() const {
  return getArch() == llvm::Triple::x86_64;
}

bool MinGW::isPICDefault() const { return getArch() == llvm::Triple::x86_64; }

bool MinGW::isPIEDefault() const { return false; }

bool MinGW::isPICDefaultForced() const {
  return getArch() == llvm::Triple::x86_64;
}

bool MinGW::UseSEHExceptions() const {
  return getArch() == llvm::Triple::x86_64;
}

void MinGW::AddCudaIncludeArgs(const ArgList &DriverArgs,
                               ArgStringList &CC1Args) const {
  CudaInstallation.AddCudaIncludeArgs(DriverArgs, CC1Args);
}

void MinGW::printVerboseInfo(raw_ostream &OS) const {
  CudaInstallation.print(OS);
}

// Include directories for various hosts:

// Windows, mingw.org
// c:\mingw\lib\gcc\mingw32\4.8.1\include\c++
// c:\mingw\lib\gcc\mingw32\4.8.1\include\c++\mingw32
// c:\mingw\lib\gcc\mingw32\4.8.1\include\c++\backward
// c:\mingw\lib\gcc\mingw32\4.8.1\include
// c:\mingw\include
// c:\mingw\lib\gcc\mingw32\4.8.1\include-fixed
// c:\mingw\mingw32\include

// Windows, mingw-w64 mingw-builds
// c:\mingw32\lib\gcc\i686-w64-mingw32\4.9.1\include
// c:\mingw32\lib\gcc\i686-w64-mingw32\4.9.1\include-fixed
// c:\mingw32\i686-w64-mingw32\include
// c:\mingw32\i686-w64-mingw32\include\c++
// c:\mingw32\i686-w64-mingw32\include\c++\i686-w64-mingw32
// c:\mingw32\i686-w64-mingw32\include\c++\backward

// Windows, mingw-w64 msys2
// c:\msys64\mingw32\lib\gcc\i686-w64-mingw32\4.9.2\include
// c:\msys64\mingw32\include
// c:\msys64\mingw32\lib\gcc\i686-w64-mingw32\4.9.2\include-fixed
// c:\msys64\mingw32\i686-w64-mingw32\include
// c:\msys64\mingw32\include\c++\4.9.2
// c:\msys64\mingw32\include\c++\4.9.2\i686-w64-mingw32
// c:\msys64\mingw32\include\c++\4.9.2\backward

// openSUSE
// /usr/lib64/gcc/x86_64-w64-mingw32/5.1.0/include/c++
// /usr/lib64/gcc/x86_64-w64-mingw32/5.1.0/include/c++/x86_64-w64-mingw32
// /usr/lib64/gcc/x86_64-w64-mingw32/5.1.0/include/c++/backward
// /usr/lib64/gcc/x86_64-w64-mingw32/5.1.0/include
// /usr/lib64/gcc/x86_64-w64-mingw32/5.1.0/include-fixed
// /usr/x86_64-w64-mingw32/sys-root/mingw/include

// Arch Linux
// /usr/i686-w64-mingw32/include/c++/5.1.0
// /usr/i686-w64-mingw32/include/c++/5.1.0/i686-w64-mingw32
// /usr/i686-w64-mingw32/include/c++/5.1.0/backward
// /usr/lib/gcc/i686-w64-mingw32/5.1.0/include
// /usr/lib/gcc/i686-w64-mingw32/5.1.0/include-fixed
// /usr/i686-w64-mingw32/include

// Ubuntu
// /usr/include/c++/4.8
// /usr/include/c++/4.8/x86_64-w64-mingw32
// /usr/include/c++/4.8/backward
// /usr/lib/gcc/x86_64-w64-mingw32/4.8/include
// /usr/lib/gcc/x86_64-w64-mingw32/4.8/include-fixed
// /usr/x86_64-w64-mingw32/include

void MinGW::AddClangSystemIncludeArgs(const ArgList &DriverArgs,
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
    llvm::SmallString<1024> IncludeDir(GccLibDir);
    llvm::sys::path::append(IncludeDir, "include");
    addSystemInclude(DriverArgs, CC1Args, IncludeDir.c_str());
    IncludeDir += "-fixed";
    // openSUSE
    addSystemInclude(DriverArgs, CC1Args,
                     Base + Arch + "/sys-root/mingw/include");
    addSystemInclude(DriverArgs, CC1Args, IncludeDir.c_str());
  }
  addSystemInclude(DriverArgs, CC1Args,
                   Base + Arch + llvm::sys::path::get_separator() + "include");
  addSystemInclude(DriverArgs, CC1Args, Base + "include");
}

void MinGW::AddClangCXXStdlibIncludeArgs(const ArgList &DriverArgs,
                                         ArgStringList &CC1Args) const {
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
