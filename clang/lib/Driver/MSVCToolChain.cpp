//===--- ToolChains.cpp - ToolChain Implementations -----------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "ToolChains.h"
#include "Tools.h"
#include "clang/Basic/CharInfo.h"
#include "clang/Basic/Version.h"
#include "clang/Driver/Compilation.h"
#include "clang/Driver/Driver.h"
#include "clang/Driver/DriverDiagnostic.h"
#include "clang/Driver/Options.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Config/llvm-config.h"
#include "llvm/Option/Arg.h"
#include "llvm/Option/ArgList.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Process.h"
#include <cstdio>

// Include the necessary headers to interface with the Windows registry and
// environment.
#if defined(LLVM_ON_WIN32)
#define USE_WIN32
#endif

#ifdef USE_WIN32
  #define WIN32_LEAN_AND_MEAN
  #define NOGDI
  #ifndef NOMINMAX
    #define NOMINMAX
  #endif
  #include <windows.h>
#endif

using namespace clang::driver;
using namespace clang::driver::toolchains;
using namespace clang;
using namespace llvm::opt;

MSVCToolChain::MSVCToolChain(const Driver &D, const llvm::Triple& Triple,
                             const ArgList &Args)
  : ToolChain(D, Triple, Args) {
  getProgramPaths().push_back(getDriver().getInstalledDir());
  if (getDriver().getInstalledDir() != getDriver().Dir)
    getProgramPaths().push_back(getDriver().Dir);
}

Tool *MSVCToolChain::buildLinker() const {
  return new tools::visualstudio::Linker(*this);
}

Tool *MSVCToolChain::buildAssembler() const {
  if (getTriple().isOSBinFormatMachO())
    return new tools::darwin::Assembler(*this);
  getDriver().Diag(clang::diag::err_no_external_assembler);
  return nullptr;
}

bool MSVCToolChain::IsIntegratedAssemblerDefault() const {
  return true;
}

bool MSVCToolChain::IsUnwindTablesDefault() const {
  // Emit unwind tables by default on Win64. All non-x86_32 Windows platforms
  // such as ARM and PPC actually require unwind tables, but LLVM doesn't know
  // how to generate them yet.
  return getArch() == llvm::Triple::x86_64;
}

bool MSVCToolChain::isPICDefault() const {
  return getArch() == llvm::Triple::x86_64;
}

bool MSVCToolChain::isPIEDefault() const {
  return false;
}

bool MSVCToolChain::isPICDefaultForced() const {
  return getArch() == llvm::Triple::x86_64;
}

#ifdef USE_WIN32
static bool readFullStringValue(HKEY hkey, const char *valueName,
                                std::string &value) {
  // FIXME: We should be using the W versions of the registry functions, but
  // doing so requires UTF8 / UTF16 conversions similar to how we handle command
  // line arguments.  The UTF8 conversion functions are not exposed publicly
  // from LLVM though, so in order to do this we will probably need to create
  // a registry abstraction in LLVMSupport that is Windows only.
  DWORD result = 0;
  DWORD valueSize = 0;
  DWORD type = 0;
  // First just query for the required size.
  result = RegQueryValueEx(hkey, valueName, NULL, &type, NULL, &valueSize);
  if (result != ERROR_SUCCESS || type != REG_SZ)
    return false;
  std::vector<BYTE> buffer(valueSize);
  result = RegQueryValueEx(hkey, valueName, NULL, NULL, &buffer[0], &valueSize);
  if (result == ERROR_SUCCESS)
    value.assign(reinterpret_cast<const char *>(buffer.data()));
  return result;
}
#endif

/// \brief Read registry string.
/// This also supports a means to look for high-versioned keys by use
/// of a $VERSION placeholder in the key path.
/// $VERSION in the key path is a placeholder for the version number,
/// causing the highest value path to be searched for and used.
/// I.e. "SOFTWARE\\Microsoft\\VisualStudio\\$VERSION".
/// There can be additional characters in the component.  Only the numeric
/// characters are compared.  This function only searches HKLM.
static bool getSystemRegistryString(const char *keyPath, const char *valueName,
                                    std::string &value, std::string *phValue) {
#ifndef USE_WIN32
  return false;
#else
  HKEY hRootKey = HKEY_LOCAL_MACHINE;
  HKEY hKey = NULL;
  long lResult;
  bool returnValue = false;

  const char *placeHolder = strstr(keyPath, "$VERSION");
  std::string bestName;
  // If we have a $VERSION placeholder, do the highest-version search.
  if (placeHolder) {
    const char *keyEnd = placeHolder - 1;
    const char *nextKey = placeHolder;
    // Find end of previous key.
    while ((keyEnd > keyPath) && (*keyEnd != '\\'))
      keyEnd--;
    // Find end of key containing $VERSION.
    while (*nextKey && (*nextKey != '\\'))
      nextKey++;
    size_t partialKeyLength = keyEnd - keyPath;
    char partialKey[256];
    if (partialKeyLength > sizeof(partialKey))
      partialKeyLength = sizeof(partialKey);
    strncpy(partialKey, keyPath, partialKeyLength);
    partialKey[partialKeyLength] = '\0';
    HKEY hTopKey = NULL;
    lResult = RegOpenKeyEx(hRootKey, partialKey, 0, KEY_READ | KEY_WOW64_32KEY,
                           &hTopKey);
    if (lResult == ERROR_SUCCESS) {
      char keyName[256];
      double bestValue = 0.0;
      DWORD index, size = sizeof(keyName) - 1;
      for (index = 0; RegEnumKeyEx(hTopKey, index, keyName, &size, NULL,
          NULL, NULL, NULL) == ERROR_SUCCESS; index++) {
        const char *sp = keyName;
        while (*sp && !isDigit(*sp))
          sp++;
        if (!*sp)
          continue;
        const char *ep = sp + 1;
        while (*ep && (isDigit(*ep) || (*ep == '.')))
          ep++;
        char numBuf[32];
        strncpy(numBuf, sp, sizeof(numBuf) - 1);
        numBuf[sizeof(numBuf) - 1] = '\0';
        double dvalue = strtod(numBuf, NULL);
        if (dvalue > bestValue) {
          // Test that InstallDir is indeed there before keeping this index.
          // Open the chosen key path remainder.
          bestName = keyName;
          // Append rest of key.
          bestName.append(nextKey);
          lResult = RegOpenKeyEx(hTopKey, bestName.c_str(), 0,
                                 KEY_READ | KEY_WOW64_32KEY, &hKey);
          if (lResult == ERROR_SUCCESS) {
            lResult = readFullStringValue(hKey, valueName, value);
            if (lResult == ERROR_SUCCESS) {
              bestValue = dvalue;
              if (phValue)
                *phValue = bestName;
              returnValue = true;
            }
            RegCloseKey(hKey);
          }
        }
        size = sizeof(keyName) - 1;
      }
      RegCloseKey(hTopKey);
    }
  } else {
    lResult =
        RegOpenKeyEx(hRootKey, keyPath, 0, KEY_READ | KEY_WOW64_32KEY, &hKey);
    if (lResult == ERROR_SUCCESS) {
      lResult = readFullStringValue(hKey, valueName, value);
      if (lResult == ERROR_SUCCESS)
        returnValue = true;
      if (phValue)
        phValue->clear();
      RegCloseKey(hKey);
    }
  }
  return returnValue;
#endif // USE_WIN32
}

// Convert LLVM's ArchType
// to the corresponding name of Windows SDK libraries subfolder
static StringRef getWindowsSDKArch(llvm::Triple::ArchType Arch) {
  switch (Arch) {
  case llvm::Triple::x86:
    return "x86";
  case llvm::Triple::x86_64:
    return "x64";
  case llvm::Triple::arm:
    return "arm";
  default:
    return "";
  }
}

// Find the most recent version of Universal CRT or Windows 10 SDK.
// vcvarsqueryregistry.bat from Visual Studio 2015 sorts entries in the include
// directory by name and uses the last one of the list.
// So we compare entry names lexicographically to find the greatest one.
static bool getWindows10SDKVersion(const std::string &SDKPath,
                                   std::string &SDKVersion) {
  SDKVersion.clear();

  std::error_code EC;
  llvm::SmallString<128> IncludePath(SDKPath);
  llvm::sys::path::append(IncludePath, "Include");
  for (llvm::sys::fs::directory_iterator DirIt(IncludePath, EC), DirEnd;
       DirIt != DirEnd && !EC; DirIt.increment(EC)) {
    if (!llvm::sys::fs::is_directory(DirIt->path()))
      continue;
    StringRef CandidateName = llvm::sys::path::filename(DirIt->path());
    // If WDK is installed, there could be subfolders like "wdf" in the
    // "Include" directory.
    // Allow only directories which names start with "10.".
    if (!CandidateName.startswith("10."))
      continue;
    if (CandidateName > SDKVersion)
      SDKVersion = CandidateName;
  }

  return !SDKVersion.empty();
}

/// \brief Get Windows SDK installation directory.
bool MSVCToolChain::getWindowsSDKDir(std::string &Path, int &Major,
                                     std::string &WindowsSDKIncludeVersion,
                                     std::string &WindowsSDKLibVersion) const {
  std::string RegistrySDKVersion;
  // Try the Windows registry.
  if (!getSystemRegistryString(
          "SOFTWARE\\Microsoft\\Microsoft SDKs\\Windows\\$VERSION",
          "InstallationFolder", Path, &RegistrySDKVersion))
    return false;
  if (Path.empty() || RegistrySDKVersion.empty())
    return false;

  WindowsSDKIncludeVersion.clear();
  WindowsSDKLibVersion.clear();
  Major = 0;
  std::sscanf(RegistrySDKVersion.c_str(), "v%d.", &Major);
  if (Major <= 7)
    return true;
  if (Major == 8) {
    // Windows SDK 8.x installs libraries in a folder whose names depend on the
    // version of the OS you're targeting.  By default choose the newest, which
    // usually corresponds to the version of the OS you've installed the SDK on.
    const char *Tests[] = {"winv6.3", "win8", "win7"};
    for (const char *Test : Tests) {
      llvm::SmallString<128> TestPath(Path);
      llvm::sys::path::append(TestPath, "Lib", Test);
      if (llvm::sys::fs::exists(TestPath.c_str())) {
        WindowsSDKLibVersion = Test;
        break;
      }
    }
    return !WindowsSDKLibVersion.empty();
  }
  if (Major == 10) {
    if (!getWindows10SDKVersion(Path, WindowsSDKIncludeVersion))
      return false;
    WindowsSDKLibVersion = WindowsSDKIncludeVersion;
    return true;
  }
  // Unsupported SDK version
  return false;
}

// Gets the library path required to link against the Windows SDK.
bool MSVCToolChain::getWindowsSDKLibraryPath(std::string &path) const {
  std::string sdkPath;
  int sdkMajor = 0;
  std::string windowsSDKIncludeVersion;
  std::string windowsSDKLibVersion;

  path.clear();
  if (!getWindowsSDKDir(sdkPath, sdkMajor, windowsSDKIncludeVersion,
                        windowsSDKLibVersion))
    return false;

  llvm::SmallString<128> libPath(sdkPath);
  llvm::sys::path::append(libPath, "Lib");
  if (sdkMajor <= 7) {
    switch (getArch()) {
    // In Windows SDK 7.x, x86 libraries are directly in the Lib folder.
    case llvm::Triple::x86:
      break;
    case llvm::Triple::x86_64:
      llvm::sys::path::append(libPath, "x64");
      break;
    case llvm::Triple::arm:
      // It is not necessary to link against Windows SDK 7.x when targeting ARM.
      return false;
    default:
      return false;
    }
  } else {
    const StringRef archName = getWindowsSDKArch(getArch());
    if (archName.empty())
      return false;
    llvm::sys::path::append(libPath, windowsSDKLibVersion, "um", archName);
  }

  path = libPath.str();
  return true;
}

// Check if the Include path of a specified version of Visual Studio contains
// specific header files. If not, they are probably shipped with Universal CRT.
bool clang::driver::toolchains::MSVCToolChain::useUniversalCRT(
    std::string &VisualStudioDir) const {
  llvm::SmallString<128> TestPath(VisualStudioDir);
  llvm::sys::path::append(TestPath, "VC\\include\\stdlib.h");

  return !llvm::sys::fs::exists(TestPath);
}

bool MSVCToolChain::getUniversalCRTSdkDir(std::string &Path,
                                          std::string &UCRTVersion) const {
  // vcvarsqueryregistry.bat for Visual Studio 2015 queries the registry
  // for the specific key "KitsRoot10". So do we.
  if (!getSystemRegistryString(
          "SOFTWARE\\Microsoft\\Windows Kits\\Installed Roots", "KitsRoot10",
          Path, nullptr))
    return false;

  return getWindows10SDKVersion(Path, UCRTVersion);
}

bool MSVCToolChain::getUniversalCRTLibraryPath(std::string &Path) const {
  std::string UniversalCRTSdkPath;
  std::string UCRTVersion;

  Path.clear();
  if (!getUniversalCRTSdkDir(UniversalCRTSdkPath, UCRTVersion))
    return false;

  StringRef ArchName = getWindowsSDKArch(getArch());
  if (ArchName.empty())
    return false;

  llvm::SmallString<128> LibPath(UniversalCRTSdkPath);
  llvm::sys::path::append(LibPath, "Lib", UCRTVersion, "ucrt", ArchName);

  Path = LibPath.str();
  return true;
}

// Get the location to use for Visual Studio binaries.  The location priority
// is: %VCINSTALLDIR% > %PATH% > newest copy of Visual Studio installed on
// system (as reported by the registry).
bool MSVCToolChain::getVisualStudioBinariesFolder(const char *clangProgramPath,
                                                  std::string &path) const {
  path.clear();

  SmallString<128> BinDir;

  // First check the environment variables that vsvars32.bat sets.
  llvm::Optional<std::string> VcInstallDir =
      llvm::sys::Process::GetEnv("VCINSTALLDIR");
  if (VcInstallDir.hasValue()) {
    BinDir = VcInstallDir.getValue();
    llvm::sys::path::append(BinDir, "bin");
  } else {
    // Next walk the PATH, trying to find a cl.exe in the path.  If we find one,
    // use that.  However, make sure it's not clang's cl.exe.
    llvm::Optional<std::string> OptPath = llvm::sys::Process::GetEnv("PATH");
    if (OptPath.hasValue()) {
      const char EnvPathSeparatorStr[] = {llvm::sys::EnvPathSeparator, '\0'};
      SmallVector<StringRef, 8> PathSegments;
      llvm::SplitString(OptPath.getValue(), PathSegments, EnvPathSeparatorStr);

      for (StringRef PathSegment : PathSegments) {
        if (PathSegment.empty())
          continue;

        SmallString<128> FilePath(PathSegment);
        llvm::sys::path::append(FilePath, "cl.exe");
        if (llvm::sys::fs::can_execute(FilePath.c_str()) &&
            !llvm::sys::fs::equivalent(FilePath.c_str(), clangProgramPath)) {
          // If we found it on the PATH, use it exactly as is with no
          // modifications.
          path = PathSegment;
          return true;
        }
      }
    }

    std::string installDir;
    // With no VCINSTALLDIR and nothing on the PATH, if we can't find it in the
    // registry then we have no choice but to fail.
    if (!getVisualStudioInstallDir(installDir))
      return false;

    // Regardless of what binary we're ultimately trying to find, we make sure
    // that this is a Visual Studio directory by checking for cl.exe.  We use
    // cl.exe instead of other binaries like link.exe because programs such as
    // GnuWin32 also have a utility called link.exe, so cl.exe is the least
    // ambiguous.
    BinDir = installDir;
    llvm::sys::path::append(BinDir, "VC", "bin");
    SmallString<128> ClPath(BinDir);
    llvm::sys::path::append(ClPath, "cl.exe");

    if (!llvm::sys::fs::can_execute(ClPath.c_str()))
      return false;
  }

  if (BinDir.empty())
    return false;

  switch (getArch()) {
  case llvm::Triple::x86:
    break;
  case llvm::Triple::x86_64:
    llvm::sys::path::append(BinDir, "amd64");
    break;
  case llvm::Triple::arm:
    llvm::sys::path::append(BinDir, "arm");
    break;
  default:
    // Whatever this is, Visual Studio doesn't have a toolchain for it.
    return false;
  }
  path = BinDir.str();
  return true;
}

// Get Visual Studio installation directory.
bool MSVCToolChain::getVisualStudioInstallDir(std::string &path) const {
  // First check the environment variables that vsvars32.bat sets.
  const char *vcinstalldir = getenv("VCINSTALLDIR");
  if (vcinstalldir) {
    path = vcinstalldir;
    path = path.substr(0, path.find("\\VC"));
    return true;
  }

  std::string vsIDEInstallDir;
  std::string vsExpressIDEInstallDir;
  // Then try the windows registry.
  bool hasVCDir =
      getSystemRegistryString("SOFTWARE\\Microsoft\\VisualStudio\\$VERSION",
                              "InstallDir", vsIDEInstallDir, nullptr);
  if (hasVCDir && !vsIDEInstallDir.empty()) {
    path = vsIDEInstallDir.substr(0, vsIDEInstallDir.find("\\Common7\\IDE"));
    return true;
  }

  bool hasVCExpressDir =
      getSystemRegistryString("SOFTWARE\\Microsoft\\VCExpress\\$VERSION",
                              "InstallDir", vsExpressIDEInstallDir, nullptr);
  if (hasVCExpressDir && !vsExpressIDEInstallDir.empty()) {
    path = vsExpressIDEInstallDir.substr(
        0, vsIDEInstallDir.find("\\Common7\\IDE"));
    return true;
  }

  // Try the environment.
  const char *vs120comntools = getenv("VS120COMNTOOLS");
  const char *vs100comntools = getenv("VS100COMNTOOLS");
  const char *vs90comntools = getenv("VS90COMNTOOLS");
  const char *vs80comntools = getenv("VS80COMNTOOLS");

  const char *vscomntools = nullptr;

  // Find any version we can
  if (vs120comntools)
    vscomntools = vs120comntools;
  else if (vs100comntools)
    vscomntools = vs100comntools;
  else if (vs90comntools)
    vscomntools = vs90comntools;
  else if (vs80comntools)
    vscomntools = vs80comntools;

  if (vscomntools && *vscomntools) {
    const char *p = strstr(vscomntools, "\\Common7\\Tools");
    path = p ? std::string(vscomntools, p) : vscomntools;
    return true;
  }
  return false;
}

void MSVCToolChain::AddSystemIncludeWithSubfolder(
    const ArgList &DriverArgs, ArgStringList &CC1Args,
    const std::string &folder, const Twine &subfolder1, const Twine &subfolder2,
    const Twine &subfolder3) const {
  llvm::SmallString<128> path(folder);
  llvm::sys::path::append(path, subfolder1, subfolder2, subfolder3);
  addSystemInclude(DriverArgs, CC1Args, path);
}

void MSVCToolChain::AddClangSystemIncludeArgs(const ArgList &DriverArgs,
                                              ArgStringList &CC1Args) const {
  if (DriverArgs.hasArg(options::OPT_nostdinc))
    return;

  if (!DriverArgs.hasArg(options::OPT_nobuiltininc)) {
    AddSystemIncludeWithSubfolder(DriverArgs, CC1Args, getDriver().ResourceDir,
                                  "include");
  }

  if (DriverArgs.hasArg(options::OPT_nostdlibinc))
    return;

  // Honor %INCLUDE%. It should know essential search paths with vcvarsall.bat.
  if (const char *cl_include_dir = getenv("INCLUDE")) {
    SmallVector<StringRef, 8> Dirs;
    StringRef(cl_include_dir)
        .split(Dirs, ";", /*MaxSplit=*/-1, /*KeepEmpty=*/false);
    for (StringRef Dir : Dirs)
      addSystemInclude(DriverArgs, CC1Args, Dir);
    if (!Dirs.empty())
      return;
  }

  std::string VSDir;

  // When built with access to the proper Windows APIs, try to actually find
  // the correct include paths first.
  if (getVisualStudioInstallDir(VSDir)) {
    AddSystemIncludeWithSubfolder(DriverArgs, CC1Args, VSDir, "VC\\include");

    if (useUniversalCRT(VSDir)) {
      std::string UniversalCRTSdkPath;
      std::string UCRTVersion;
      if (getUniversalCRTSdkDir(UniversalCRTSdkPath, UCRTVersion)) {
        AddSystemIncludeWithSubfolder(DriverArgs, CC1Args, UniversalCRTSdkPath,
                                      "Include", UCRTVersion, "ucrt");
      }
    }

    std::string WindowsSDKDir;
    int major;
    std::string windowsSDKIncludeVersion;
    std::string windowsSDKLibVersion;
    if (getWindowsSDKDir(WindowsSDKDir, major, windowsSDKIncludeVersion,
                         windowsSDKLibVersion)) {
      if (major >= 8) {
        // Note: windowsSDKIncludeVersion is empty for SDKs prior to v10.
        // Anyway, llvm::sys::path::append is able to manage it.
        AddSystemIncludeWithSubfolder(DriverArgs, CC1Args, WindowsSDKDir,
                                      "include", windowsSDKIncludeVersion,
                                      "shared");
        AddSystemIncludeWithSubfolder(DriverArgs, CC1Args, WindowsSDKDir,
                                      "include", windowsSDKIncludeVersion,
                                      "um");
        AddSystemIncludeWithSubfolder(DriverArgs, CC1Args, WindowsSDKDir,
                                      "include", windowsSDKIncludeVersion,
                                      "winrt");
      } else {
        AddSystemIncludeWithSubfolder(DriverArgs, CC1Args, WindowsSDKDir,
                                      "include");
      }
    } else {
      addSystemInclude(DriverArgs, CC1Args, VSDir);
    }
    return;
  }

  // As a fallback, select default install paths.
  // FIXME: Don't guess drives and paths like this on Windows.
  const StringRef Paths[] = {
    "C:/Program Files/Microsoft Visual Studio 10.0/VC/include",
    "C:/Program Files/Microsoft Visual Studio 9.0/VC/include",
    "C:/Program Files/Microsoft Visual Studio 9.0/VC/PlatformSDK/Include",
    "C:/Program Files/Microsoft Visual Studio 8/VC/include",
    "C:/Program Files/Microsoft Visual Studio 8/VC/PlatformSDK/Include"
  };
  addSystemIncludes(DriverArgs, CC1Args, Paths);
}

void MSVCToolChain::AddClangCXXStdlibIncludeArgs(const ArgList &DriverArgs,
                                                 ArgStringList &CC1Args) const {
  // FIXME: There should probably be logic here to find libc++ on Windows.
}

std::string
MSVCToolChain::ComputeEffectiveClangTriple(const ArgList &Args,
                                           types::ID InputType) const {
  std::string TripleStr =
      ToolChain::ComputeEffectiveClangTriple(Args, InputType);
  llvm::Triple Triple(TripleStr);
  VersionTuple MSVT =
      tools::visualstudio::getMSVCVersion(/*D=*/nullptr, Triple, Args,
                                          /*IsWindowsMSVC=*/true);
  if (MSVT.empty())
    return TripleStr;

  MSVT = VersionTuple(MSVT.getMajor(), MSVT.getMinor().getValueOr(0),
                      MSVT.getSubminor().getValueOr(0));

  if (Triple.getEnvironment() == llvm::Triple::MSVC) {
    StringRef ObjFmt = Triple.getEnvironmentName().split('-').second;
    if (ObjFmt.empty())
      Triple.setEnvironmentName((Twine("msvc") + MSVT.getAsString()).str());
    else
      Triple.setEnvironmentName(
          (Twine("msvc") + MSVT.getAsString() + Twine('-') + ObjFmt).str());
  }
  return Triple.getTriple();
}

SanitizerMask MSVCToolChain::getSupportedSanitizers() const {
  SanitizerMask Res = ToolChain::getSupportedSanitizers();
  Res |= SanitizerKind::Address;
  return Res;
}

llvm::opt::DerivedArgList *
MSVCToolChain::TranslateArgs(const llvm::opt::DerivedArgList &Args,
                             const char *BoundArch) const {
  DerivedArgList *DAL = new DerivedArgList(Args.getBaseArgs());
  const OptTable &Opts = getDriver().getOpts();

  // /Oy and /Oy- only has an effect under X86-32.
  bool SupportsForcingFramePointer = getArch() == llvm::Triple::x86;

  // The -O[12xd] flag actually expands to several flags.  We must desugar the
  // flags so that options embedded can be negated.  For example, the '-O2' flag
  // enables '-Oy'.  Expanding '-O2' into its constituent flags allows us to
  // correctly handle '-O2 -Oy-' where the trailing '-Oy-' disables a single
  // aspect of '-O2'.
  //
  // Note that this expansion logic only applies to the *last* of '[12xd]'.

  // First step is to search for the character we'd like to expand.
  const char *ExpandChar = nullptr;
  for (Arg *A : Args) {
    if (!A->getOption().matches(options::OPT__SLASH_O))
      continue;
    StringRef OptStr = A->getValue();
    for (size_t I = 0, E = OptStr.size(); I != E; ++I) {
      const char &OptChar = *(OptStr.data() + I);
      if (OptChar == '1' || OptChar == '2' || OptChar == 'x' || OptChar == 'd')
        ExpandChar = OptStr.data() + I;
    }
  }

  // The -O flag actually takes an amalgam of other options.  For example,
  // '/Ogyb2' is equivalent to '/Og' '/Oy' '/Ob2'.
  for (Arg *A : Args) {
    if (!A->getOption().matches(options::OPT__SLASH_O)) {
      DAL->append(A);
      continue;
    }

    StringRef OptStr = A->getValue();
    for (size_t I = 0, E = OptStr.size(); I != E; ++I) {
      const char &OptChar = *(OptStr.data() + I);
      switch (OptChar) {
      default:
        break;
      case '1':
      case '2':
      case 'x':
      case 'd':
        if (&OptChar == ExpandChar) {
          if (OptChar == 'd') {
            DAL->AddFlagArg(A, Opts.getOption(options::OPT_O0));
          } else {
            if (OptChar == '1') {
              DAL->AddJoinedArg(A, Opts.getOption(options::OPT_O), "s");
            } else if (OptChar == '2' || OptChar == 'x') {
              DAL->AddFlagArg(A, Opts.getOption(options::OPT_fbuiltin));
              DAL->AddJoinedArg(A, Opts.getOption(options::OPT_O), "2");
            }
            if (SupportsForcingFramePointer)
              DAL->AddFlagArg(A,
                              Opts.getOption(options::OPT_fomit_frame_pointer));
            if (OptChar == '1' || OptChar == '2')
              DAL->AddFlagArg(A,
                              Opts.getOption(options::OPT_ffunction_sections));
          }
        }
        break;
      case 'b':
        if (I + 1 != E && isdigit(OptStr[I + 1]))
          ++I;
        break;
      case 'g':
        break;
      case 'i':
        if (I + 1 != E && OptStr[I + 1] == '-') {
          ++I;
          DAL->AddFlagArg(A, Opts.getOption(options::OPT_fno_builtin));
        } else {
          DAL->AddFlagArg(A, Opts.getOption(options::OPT_fbuiltin));
        }
        break;
      case 's':
        DAL->AddJoinedArg(A, Opts.getOption(options::OPT_O), "s");
        break;
      case 't':
        DAL->AddJoinedArg(A, Opts.getOption(options::OPT_O), "2");
        break;
      case 'y': {
        bool OmitFramePointer = true;
        if (I + 1 != E && OptStr[I + 1] == '-') {
          OmitFramePointer = false;
          ++I;
        }
        if (SupportsForcingFramePointer) {
          if (OmitFramePointer)
            DAL->AddFlagArg(A,
                            Opts.getOption(options::OPT_fomit_frame_pointer));
          else
            DAL->AddFlagArg(
                A, Opts.getOption(options::OPT_fno_omit_frame_pointer));
        }
        break;
      }
      }
    }
  }
  return DAL;
}
