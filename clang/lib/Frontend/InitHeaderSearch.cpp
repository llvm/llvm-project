//===--- InitHeaderSearch.cpp - Initialize header search paths ------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the InitHeaderSearch class.
//
//===----------------------------------------------------------------------===//

#ifdef HAVE_CLANG_CONFIG_H
# include "clang/Config/config.h"
#endif

#include "clang/Frontend/Utils.h"
#include "clang/Basic/FileManager.h"
#include "clang/Basic/LangOptions.h"
#include "clang/Frontend/HeaderSearchOptions.h"
#include "clang/Lex/HeaderSearch.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/Triple.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/Path.h"
#include "llvm/Config/config.h"
#ifdef _MSC_VER
  #define WIN32_LEAN_AND_MEAN 1
  #include <windows.h>
#endif
using namespace clang;
using namespace clang::frontend;

namespace {

/// InitHeaderSearch - This class makes it easier to set the search paths of
///  a HeaderSearch object. InitHeaderSearch stores several search path lists
///  internally, which can be sent to a HeaderSearch object in one swoop.
class InitHeaderSearch {
  std::vector<std::pair<IncludeDirGroup, DirectoryLookup> > IncludePath;
  typedef std::vector<std::pair<IncludeDirGroup,
                      DirectoryLookup> >::const_iterator path_iterator;
  HeaderSearch &Headers;
  bool Verbose;
  std::string IncludeSysroot;
  bool IsNotEmptyOrRoot;

public:

  InitHeaderSearch(HeaderSearch &HS, bool verbose, StringRef sysroot)
    : Headers(HS), Verbose(verbose), IncludeSysroot(sysroot),
      IsNotEmptyOrRoot(!(sysroot.empty() || sysroot == "/")) {
  }

  /// AddPath - Add the specified path to the specified group list.
  void AddPath(const Twine &Path, IncludeDirGroup Group,
               bool isCXXAware, bool isUserSupplied,
               bool isFramework, bool IgnoreSysRoot = false);

  /// AddGnuCPlusPlusIncludePaths - Add the necessary paths to support a gnu
  ///  libstdc++.
  void AddGnuCPlusPlusIncludePaths(StringRef Base,
                                   StringRef ArchDir,
                                   StringRef Dir32,
                                   StringRef Dir64,
                                   const llvm::Triple &triple);

  /// AddMinGWCPlusPlusIncludePaths - Add the necessary paths to support a MinGW
  ///  libstdc++.
  void AddMinGWCPlusPlusIncludePaths(StringRef Base,
                                     StringRef Arch,
                                     StringRef Version);

  /// AddMinGW64CXXPaths - Add the necessary paths to support
  /// libstdc++ of x86_64-w64-mingw32 aka mingw-w64.
  void AddMinGW64CXXPaths(StringRef Base,
                          StringRef Version);

  // AddDefaultCIncludePaths - Add paths that should always be searched.
  void AddDefaultCIncludePaths(const llvm::Triple &triple,
                               const HeaderSearchOptions &HSOpts);

  // AddDefaultCPlusPlusIncludePaths -  Add paths that should be searched when
  //  compiling c++.
  void AddDefaultCPlusPlusIncludePaths(const llvm::Triple &triple,
                                       const HeaderSearchOptions &HSOpts);

  /// AddDefaultSystemIncludePaths - Adds the default system include paths so
  ///  that e.g. stdio.h is found.
  void AddDefaultIncludePaths(const LangOptions &Lang,
                              const llvm::Triple &triple,
                              const HeaderSearchOptions &HSOpts);

  /// Realize - Merges all search path lists into one list and send it to
  /// HeaderSearch.
  void Realize(const LangOptions &Lang);
};

}  // end anonymous namespace.

void InitHeaderSearch::AddPath(const Twine &Path,
                               IncludeDirGroup Group, bool isCXXAware,
                               bool isUserSupplied, bool isFramework,
                               bool IgnoreSysRoot) {
  assert(!Path.isTriviallyEmpty() && "can't handle empty path here");
  FileManager &FM = Headers.getFileMgr();

  // Compute the actual path, taking into consideration -isysroot.
  llvm::SmallString<256> MappedPathStorage;
  StringRef MappedPathStr = Path.toStringRef(MappedPathStorage);

  // Handle isysroot.
  if ((Group == System || Group == CXXSystem) && !IgnoreSysRoot &&
#if defined(_WIN32)
      !MappedPathStr.empty() &&
      llvm::sys::path::is_separator(MappedPathStr[0]) &&
#else
      llvm::sys::path::is_absolute(MappedPathStr) &&
#endif
      IsNotEmptyOrRoot) {
    MappedPathStorage.clear();
    MappedPathStr =
      (IncludeSysroot + Path).toStringRef(MappedPathStorage);
  }

  // Compute the DirectoryLookup type.
  SrcMgr::CharacteristicKind Type;
  if (Group == Quoted || Group == Angled || Group == IndexHeaderMap)
    Type = SrcMgr::C_User;
  else if (isCXXAware)
    Type = SrcMgr::C_System;
  else
    Type = SrcMgr::C_ExternCSystem;


  // If the directory exists, add it.
  if (const DirectoryEntry *DE = FM.getDirectory(MappedPathStr)) {
    IncludePath.push_back(std::make_pair(Group, DirectoryLookup(DE, Type,
                          isUserSupplied, isFramework)));
    return;
  }

  // Check to see if this is an apple-style headermap (which are not allowed to
  // be frameworks).
  if (!isFramework) {
    if (const FileEntry *FE = FM.getFile(MappedPathStr)) {
      if (const HeaderMap *HM = Headers.CreateHeaderMap(FE)) {
        // It is a headermap, add it to the search path.
        IncludePath.push_back(std::make_pair(Group, DirectoryLookup(HM, Type,
                              isUserSupplied, Group == IndexHeaderMap)));
        return;
      }
    }
  }

  if (Verbose)
    llvm::errs() << "ignoring nonexistent directory \""
                 << MappedPathStr << "\"\n";
}

void InitHeaderSearch::AddGnuCPlusPlusIncludePaths(StringRef Base,
                                                   StringRef ArchDir,
                                                   StringRef Dir32,
                                                   StringRef Dir64,
                                                   const llvm::Triple &triple) {
  // Add the base dir
  AddPath(Base, CXXSystem, true, false, false);

  // Add the multilib dirs
  llvm::Triple::ArchType arch = triple.getArch();
  bool is64bit = arch == llvm::Triple::ppc64 || arch == llvm::Triple::x86_64;
  if (is64bit)
    AddPath(Base + "/" + ArchDir + "/" + Dir64, CXXSystem, true, false, false);
  else
    AddPath(Base + "/" + ArchDir + "/" + Dir32, CXXSystem, true, false, false);

  // Add the backward dir
  AddPath(Base + "/backward", CXXSystem, true, false, false);
}

void InitHeaderSearch::AddMinGWCPlusPlusIncludePaths(StringRef Base,
                                                     StringRef Arch,
                                                     StringRef Version) {
  AddPath(Base + "/" + Arch + "/" + Version + "/include/c++",
          CXXSystem, true, false, false);
  AddPath(Base + "/" + Arch + "/" + Version + "/include/c++/" + Arch,
          CXXSystem, true, false, false);
  AddPath(Base + "/" + Arch + "/" + Version + "/include/c++/backward",
          CXXSystem, true, false, false);
}

void InitHeaderSearch::AddMinGW64CXXPaths(StringRef Base,
                                          StringRef Version) {
  // Assumes Base is HeaderSearchOpts' ResourceDir
  AddPath(Base + "/../../../include/c++/" + Version,
          CXXSystem, true, false, false);
  AddPath(Base + "/../../../include/c++/" + Version + "/x86_64-w64-mingw32",
          CXXSystem, true, false, false);
  AddPath(Base + "/../../../include/c++/" + Version + "/i686-w64-mingw32",
          CXXSystem, true, false, false);
  AddPath(Base + "/../../../include/c++/" + Version + "/backward",
          CXXSystem, true, false, false);
}

  // FIXME: This probably should goto to some platform utils place.
#ifdef _MSC_VER

  // Read registry string.
  // This also supports a means to look for high-versioned keys by use
  // of a $VERSION placeholder in the key path.
  // $VERSION in the key path is a placeholder for the version number,
  // causing the highest value path to be searched for and used.
  // I.e. "HKEY_LOCAL_MACHINE\\SOFTWARE\\Microsoft\\VisualStudio\\$VERSION".
  // There can be additional characters in the component.  Only the numberic
  // characters are compared.
static bool getSystemRegistryString(const char *keyPath, const char *valueName,
                                    char *value, size_t maxLength) {
  HKEY hRootKey = NULL;
  HKEY hKey = NULL;
  const char* subKey = NULL;
  DWORD valueType;
  DWORD valueSize = maxLength - 1;
  long lResult;
  bool returnValue = false;
  
  if (strncmp(keyPath, "HKEY_CLASSES_ROOT\\", 18) == 0) {
    hRootKey = HKEY_CLASSES_ROOT;
    subKey = keyPath + 18;
  } else if (strncmp(keyPath, "HKEY_USERS\\", 11) == 0) {
    hRootKey = HKEY_USERS;
    subKey = keyPath + 11;
  } else if (strncmp(keyPath, "HKEY_LOCAL_MACHINE\\", 19) == 0) {
    hRootKey = HKEY_LOCAL_MACHINE;
    subKey = keyPath + 19;
  } else if (strncmp(keyPath, "HKEY_CURRENT_USER\\", 18) == 0) {
    hRootKey = HKEY_CURRENT_USER;
    subKey = keyPath + 18;
  }
  else
    return false;
  
  const char *placeHolder = strstr(subKey, "$VERSION");
  char bestName[256];
  bestName[0] = '\0';
  // If we have a $VERSION placeholder, do the highest-version search.
  if (placeHolder) {
    const char *keyEnd = placeHolder - 1;
    const char *nextKey = placeHolder;
    // Find end of previous key.
    while ((keyEnd > subKey) && (*keyEnd != '\\'))
      keyEnd--;
    // Find end of key containing $VERSION.
    while (*nextKey && (*nextKey != '\\'))
      nextKey++;
    size_t partialKeyLength = keyEnd - subKey;
    char partialKey[256];
    if (partialKeyLength > sizeof(partialKey))
      partialKeyLength = sizeof(partialKey);
    strncpy(partialKey, subKey, partialKeyLength);
    partialKey[partialKeyLength] = '\0';
    HKEY hTopKey = NULL;
    lResult = RegOpenKeyEx(hRootKey, partialKey, 0, KEY_READ, &hTopKey);
    if (lResult == ERROR_SUCCESS) {
      char keyName[256];
      int bestIndex = -1;
      double bestValue = 0.0;
      DWORD index, size = sizeof(keyName) - 1;
      for (index = 0; RegEnumKeyEx(hTopKey, index, keyName, &size, NULL,
          NULL, NULL, NULL) == ERROR_SUCCESS; index++) {
        const char *sp = keyName;
        while (*sp && !isdigit(*sp))
          sp++;
        if (!*sp)
          continue;
        const char *ep = sp + 1;
        while (*ep && (isdigit(*ep) || (*ep == '.')))
          ep++;
        char numBuf[32];
        strncpy(numBuf, sp, sizeof(numBuf) - 1);
        numBuf[sizeof(numBuf) - 1] = '\0';
        double value = strtod(numBuf, NULL);
        if (value > bestValue) {
          bestIndex = (int)index;
          bestValue = value;
          strcpy(bestName, keyName);
        }
        size = sizeof(keyName) - 1;
      }
      // If we found the highest versioned key, open the key and get the value.
      if (bestIndex != -1) {
        // Append rest of key.
        strncat(bestName, nextKey, sizeof(bestName) - 1);
        bestName[sizeof(bestName) - 1] = '\0';
        // Open the chosen key path remainder.
        lResult = RegOpenKeyEx(hTopKey, bestName, 0, KEY_READ, &hKey);
        if (lResult == ERROR_SUCCESS) {
          lResult = RegQueryValueEx(hKey, valueName, NULL, &valueType,
            (LPBYTE)value, &valueSize);
          if (lResult == ERROR_SUCCESS)
            returnValue = true;
          RegCloseKey(hKey);
        }
      }
      RegCloseKey(hTopKey);
    }
  }
  else {
    lResult = RegOpenKeyEx(hRootKey, subKey, 0, KEY_READ, &hKey);
    if (lResult == ERROR_SUCCESS) {
      lResult = RegQueryValueEx(hKey, valueName, NULL, &valueType,
        (LPBYTE)value, &valueSize);
      if (lResult == ERROR_SUCCESS)
        returnValue = true;
      RegCloseKey(hKey);
    }
  }
  return returnValue;
}
#else // _MSC_VER
  // Read registry string.
static bool getSystemRegistryString(const char*, const char*, char*, size_t) {
  return(false);
}
#endif // _MSC_VER

  // Get Visual Studio installation directory.
static bool getVisualStudioDir(std::string &path) {
  // First check the environment variables that vsvars32.bat sets.
  const char* vcinstalldir = getenv("VCINSTALLDIR");
  if (vcinstalldir) {
    char *p = const_cast<char *>(strstr(vcinstalldir, "\\VC"));
    if (p)
      *p = '\0';
    path = vcinstalldir;
    return true;
  }

  char vsIDEInstallDir[256];
  char vsExpressIDEInstallDir[256];
  // Then try the windows registry.
  bool hasVCDir = getSystemRegistryString(
    "HKEY_LOCAL_MACHINE\\SOFTWARE\\Microsoft\\VisualStudio\\$VERSION",
    "InstallDir", vsIDEInstallDir, sizeof(vsIDEInstallDir) - 1);
  bool hasVCExpressDir = getSystemRegistryString(
    "HKEY_LOCAL_MACHINE\\SOFTWARE\\Microsoft\\VCExpress\\$VERSION",
    "InstallDir", vsExpressIDEInstallDir, sizeof(vsExpressIDEInstallDir) - 1);
    // If we have both vc80 and vc90, pick version we were compiled with.
  if (hasVCDir && vsIDEInstallDir[0]) {
    char *p = (char*)strstr(vsIDEInstallDir, "\\Common7\\IDE");
    if (p)
      *p = '\0';
    path = vsIDEInstallDir;
    return true;
  }
  
  if (hasVCExpressDir && vsExpressIDEInstallDir[0]) {
    char *p = (char*)strstr(vsExpressIDEInstallDir, "\\Common7\\IDE");
    if (p)
      *p = '\0';
    path = vsExpressIDEInstallDir;
    return true;
  }

  // Try the environment.
  const char *vs100comntools = getenv("VS100COMNTOOLS");
  const char *vs90comntools = getenv("VS90COMNTOOLS");
  const char *vs80comntools = getenv("VS80COMNTOOLS");
  const char *vscomntools = NULL;

  // Try to find the version that we were compiled with
  if(false) {}
  #if (_MSC_VER >= 1600)  // VC100
  else if(vs100comntools) {
    vscomntools = vs100comntools;
  }
  #elif (_MSC_VER == 1500) // VC80
  else if(vs90comntools) {
    vscomntools = vs90comntools;
  }
  #elif (_MSC_VER == 1400) // VC80
  else if(vs80comntools) {
    vscomntools = vs80comntools;
  }
  #endif
  // Otherwise find any version we can
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

  // Get Windows SDK installation directory.
static bool getWindowsSDKDir(std::string &path) {
  char windowsSDKInstallDir[256];
  // Try the Windows registry.
  bool hasSDKDir = getSystemRegistryString(
   "HKEY_LOCAL_MACHINE\\SOFTWARE\\Microsoft\\Microsoft SDKs\\Windows\\$VERSION",
                                           "InstallationFolder",
                                           windowsSDKInstallDir,
                                           sizeof(windowsSDKInstallDir) - 1);
    // If we have both vc80 and vc90, pick version we were compiled with.
  if (hasSDKDir && windowsSDKInstallDir[0]) {
    path = windowsSDKInstallDir;
    return(true);
  }
  return(false);
}

void InitHeaderSearch::AddDefaultCIncludePaths(const llvm::Triple &triple,
                                            const HeaderSearchOptions &HSOpts) {
  llvm::Triple::OSType os = triple.getOS();

  if (HSOpts.UseStandardSystemIncludes) {
    switch (os) {
    case llvm::Triple::FreeBSD:
    case llvm::Triple::NetBSD:
      break;
    default:
      // FIXME: temporary hack: hard-coded paths.
      AddPath("/usr/local/include", System, true, false, false);
      break;
    }
  }

  // Builtin includes use #include_next directives and should be positioned
  // just prior C include dirs.
  if (HSOpts.UseBuiltinIncludes) {
    // Ignore the sys root, we *always* look for clang headers relative to
    // supplied path.
    llvm::sys::Path P(HSOpts.ResourceDir);
    P.appendComponent("include");
    AddPath(P.str(), System, false, false, false, /*IgnoreSysRoot=*/ true);
  }

  // All remaining additions are for system include directories, early exit if
  // we aren't using them.
  if (!HSOpts.UseStandardSystemIncludes)
    return;

  // Add dirs specified via 'configure --with-c-include-dirs'.
  StringRef CIncludeDirs(C_INCLUDE_DIRS);
  if (CIncludeDirs != "") {
    SmallVector<StringRef, 5> dirs;
    CIncludeDirs.split(dirs, ":");
    for (SmallVectorImpl<StringRef>::iterator i = dirs.begin();
         i != dirs.end();
         ++i)
      AddPath(*i, System, false, false, false);
    return;
  }

  switch (os) {
  case llvm::Triple::Win32: {
    std::string VSDir;
    std::string WindowsSDKDir;
    if (getVisualStudioDir(VSDir)) {
      AddPath(VSDir + "\\VC\\include", System, false, false, false);
      if (getWindowsSDKDir(WindowsSDKDir))
        AddPath(WindowsSDKDir + "\\include", System, false, false, false);
      else
        AddPath(VSDir + "\\VC\\PlatformSDK\\Include",
                System, false, false, false);
    } else {
      // Default install paths.
      AddPath("C:/Program Files/Microsoft Visual Studio 10.0/VC/include",
              System, false, false, false);
      AddPath("C:/Program Files/Microsoft Visual Studio 9.0/VC/include",
              System, false, false, false);
      AddPath(
        "C:/Program Files/Microsoft Visual Studio 9.0/VC/PlatformSDK/Include",
        System, false, false, false);
      AddPath("C:/Program Files/Microsoft Visual Studio 8/VC/include",
              System, false, false, false);
      AddPath(
        "C:/Program Files/Microsoft Visual Studio 8/VC/PlatformSDK/Include",
        System, false, false, false);
    }
    break;
  }
  case llvm::Triple::Haiku:
    AddPath("/boot/common/include", System, true, false, false);
    AddPath("/boot/develop/headers/os", System, true, false, false);
    AddPath("/boot/develop/headers/os/app", System, true, false, false);
    AddPath("/boot/develop/headers/os/arch", System, true, false, false);
    AddPath("/boot/develop/headers/os/device", System, true, false, false);
    AddPath("/boot/develop/headers/os/drivers", System, true, false, false);
    AddPath("/boot/develop/headers/os/game", System, true, false, false);
    AddPath("/boot/develop/headers/os/interface", System, true, false, false);
    AddPath("/boot/develop/headers/os/kernel", System, true, false, false);
    AddPath("/boot/develop/headers/os/locale", System, true, false, false);
    AddPath("/boot/develop/headers/os/mail", System, true, false, false);
    AddPath("/boot/develop/headers/os/media", System, true, false, false);
    AddPath("/boot/develop/headers/os/midi", System, true, false, false);
    AddPath("/boot/develop/headers/os/midi2", System, true, false, false);
    AddPath("/boot/develop/headers/os/net", System, true, false, false);
    AddPath("/boot/develop/headers/os/storage", System, true, false, false);
    AddPath("/boot/develop/headers/os/support", System, true, false, false);
    AddPath("/boot/develop/headers/os/translation",
      System, true, false, false);
    AddPath("/boot/develop/headers/os/add-ons/graphics",
      System, true, false, false);
    AddPath("/boot/develop/headers/os/add-ons/input_server",
      System, true, false, false);
    AddPath("/boot/develop/headers/os/add-ons/screen_saver",
      System, true, false, false);
    AddPath("/boot/develop/headers/os/add-ons/tracker",
      System, true, false, false);
    AddPath("/boot/develop/headers/os/be_apps/Deskbar",
      System, true, false, false);
    AddPath("/boot/develop/headers/os/be_apps/NetPositive",
      System, true, false, false);
    AddPath("/boot/develop/headers/os/be_apps/Tracker",
      System, true, false, false);
    AddPath("/boot/develop/headers/cpp", System, true, false, false);
    AddPath("/boot/develop/headers/cpp/i586-pc-haiku",
      System, true, false, false);
    AddPath("/boot/develop/headers/3rdparty", System, true, false, false);
    AddPath("/boot/develop/headers/bsd", System, true, false, false);
    AddPath("/boot/develop/headers/glibc", System, true, false, false);
    AddPath("/boot/develop/headers/posix", System, true, false, false);
    AddPath("/boot/develop/headers",  System, true, false, false);
    break;
  case llvm::Triple::RTEMS:
    break;
  case llvm::Triple::Cygwin:
    AddPath("/usr/include/w32api", System, true, false, false);
    break;
  case llvm::Triple::MinGW32: { 
      // mingw-w64 crt include paths
      llvm::sys::Path P(HSOpts.ResourceDir);
      P.appendComponent("../../../i686-w64-mingw32/include"); // <sysroot>/i686-w64-mingw32/include
      AddPath(P.str(), System, true, false, false);
      P = llvm::sys::Path(HSOpts.ResourceDir);
      P.appendComponent("../../../x86_64-w64-mingw32/include"); // <sysroot>/x86_64-w64-mingw32/include
      AddPath(P.str(), System, true, false, false);
      // mingw.org crt include paths
      P = llvm::sys::Path(HSOpts.ResourceDir);
      P.appendComponent("../../../include"); // <sysroot>/include
      AddPath(P.str(), System, true, false, false);
      AddPath("/mingw/include", System, true, false, false);
      AddPath("c:/mingw/include", System, true, false, false); 
    }
    break;
      
  case llvm::Triple::Linux:
    // Generic Debian multiarch support:
    if (triple.getArch() == llvm::Triple::x86_64) {
      AddPath("/usr/include/x86_64-linux-gnu", System, false, false, false);
      AddPath("/usr/include/i686-linux-gnu/64", System, false, false, false);
      AddPath("/usr/include/i486-linux-gnu/64", System, false, false, false);
    } else if (triple.getArch() == llvm::Triple::x86) {
      AddPath("/usr/include/x86_64-linux-gnu/32", System, false, false, false);
      AddPath("/usr/include/i686-linux-gnu", System, false, false, false);
      AddPath("/usr/include/i486-linux-gnu", System, false, false, false);
      AddPath("/usr/include/i386-linux-gnu", System, false, false, false);
    } else if (triple.getArch() == llvm::Triple::arm) {
      AddPath("/usr/include/arm-linux-gnueabi", System, false, false, false);
    }
  default:
    break;
  }

  if ( os != llvm::Triple::RTEMS )
    AddPath("/usr/include", System, false, false, false);
}

void InitHeaderSearch::
AddDefaultCPlusPlusIncludePaths(const llvm::Triple &triple, const HeaderSearchOptions &HSOpts) {
  llvm::Triple::OSType os = triple.getOS();
  StringRef CxxIncludeRoot(CXX_INCLUDE_ROOT);
  if (CxxIncludeRoot != "") {
    StringRef CxxIncludeArch(CXX_INCLUDE_ARCH);
    if (CxxIncludeArch == "")
      AddGnuCPlusPlusIncludePaths(CxxIncludeRoot, triple.str().c_str(),
                                  CXX_INCLUDE_32BIT_DIR, CXX_INCLUDE_64BIT_DIR,
                                  triple);
    else
      AddGnuCPlusPlusIncludePaths(CxxIncludeRoot, CXX_INCLUDE_ARCH,
                                  CXX_INCLUDE_32BIT_DIR, CXX_INCLUDE_64BIT_DIR,
                                  triple);
    return;
  }
  // FIXME: temporary hack: hard-coded paths.

  if (triple.isOSDarwin()) {
    switch (triple.getArch()) {
    default: break;

    case llvm::Triple::ppc:
    case llvm::Triple::ppc64:
      AddGnuCPlusPlusIncludePaths("/usr/include/c++/4.2.1",
                                  "powerpc-apple-darwin10", "", "ppc64",
                                  triple);
      AddGnuCPlusPlusIncludePaths("/usr/include/c++/4.0.0",
                                  "powerpc-apple-darwin10", "", "ppc64",
                                  triple);
      break;

    case llvm::Triple::x86:
    case llvm::Triple::x86_64:
      AddGnuCPlusPlusIncludePaths("/usr/include/c++/4.2.1",
                                  "i686-apple-darwin10", "", "x86_64", triple);
      AddGnuCPlusPlusIncludePaths("/usr/include/c++/4.0.0",
                                  "i686-apple-darwin8", "", "", triple);
      break;

    case llvm::Triple::arm:
    case llvm::Triple::thumb:
      AddGnuCPlusPlusIncludePaths("/usr/include/c++/4.2.1",
                                  "arm-apple-darwin10", "v7", "", triple);
      AddGnuCPlusPlusIncludePaths("/usr/include/c++/4.2.1",
                                  "arm-apple-darwin10", "v6", "", triple);
      break;
    }
    return;
  }

  switch (os) {
  case llvm::Triple::Cygwin:
    // Cygwin-1.7
    AddMinGWCPlusPlusIncludePaths("/usr/lib/gcc", "i686-pc-cygwin", "4.3.4");
    // g++-4 / Cygwin-1.5
    AddMinGWCPlusPlusIncludePaths("/usr/lib/gcc", "i686-pc-cygwin", "4.3.2");
    // FIXME: Do we support g++-3.4.4?
    AddMinGWCPlusPlusIncludePaths("/usr/lib/gcc", "i686-pc-cygwin", "3.4.4");
    break;
  case llvm::Triple::MinGW32:
    // mingw-w64 C++ include paths (i686-w64-mingw32 and x86_64-w64-mingw32)
    AddMinGW64CXXPaths(HSOpts.ResourceDir, "4.5.0");
    AddMinGW64CXXPaths(HSOpts.ResourceDir, "4.5.1");
    AddMinGW64CXXPaths(HSOpts.ResourceDir, "4.5.2");
    AddMinGW64CXXPaths(HSOpts.ResourceDir, "4.5.3");
    AddMinGW64CXXPaths(HSOpts.ResourceDir, "4.6.0");
    AddMinGW64CXXPaths(HSOpts.ResourceDir, "4.6.1");
    AddMinGW64CXXPaths(HSOpts.ResourceDir, "4.6.2");
    AddMinGW64CXXPaths(HSOpts.ResourceDir, "4.7.0");
    // mingw.org C++ include paths
    AddMinGWCPlusPlusIncludePaths("/mingw/lib/gcc", "mingw32", "4.5.2"); //MSYS
    AddMinGWCPlusPlusIncludePaths("c:/MinGW/lib/gcc", "mingw32", "4.5.0");
    AddMinGWCPlusPlusIncludePaths("c:/MinGW/lib/gcc", "mingw32", "4.4.0");
    AddMinGWCPlusPlusIncludePaths("c:/MinGW/lib/gcc", "mingw32", "4.3.0");
    break;
  case llvm::Triple::DragonFly:
    AddPath("/usr/include/c++/4.1", CXXSystem, true, false, false);
    break;
  case llvm::Triple::Linux:
    //===------------------------------------------------------------------===//
    // Debian based distros.
    // Note: these distros symlink /usr/include/c++/X.Y.Z -> X.Y
    //===------------------------------------------------------------------===//

    // Ubuntu 11.11 "Oneiric Ocelot" -- gcc-4.6.0
    AddGnuCPlusPlusIncludePaths("/usr/include/c++/4.6",
                                "x86_64-linux-gnu", "32", "", triple);
    AddGnuCPlusPlusIncludePaths("/usr/include/c++/4.6",
                                "i686-linux-gnu", "", "64", triple);
    AddGnuCPlusPlusIncludePaths("/usr/include/c++/4.6",
                                "i486-linux-gnu", "", "64", triple);
    AddGnuCPlusPlusIncludePaths("/usr/include/c++/4.6",
                                "arm-linux-gnueabi", "", "", triple);

    // Ubuntu 11.04 "Natty Narwhal" -- gcc-4.5.2
    AddGnuCPlusPlusIncludePaths("/usr/include/c++/4.5",
                                "x86_64-linux-gnu", "32", "", triple);
    AddGnuCPlusPlusIncludePaths("/usr/include/c++/4.5",
                                "i686-linux-gnu", "", "64", triple);
    AddGnuCPlusPlusIncludePaths("/usr/include/c++/4.5",
                                "i486-linux-gnu", "", "64", triple);
    AddGnuCPlusPlusIncludePaths("/usr/include/c++/4.5",
                                "arm-linux-gnueabi", "", "", triple);

    // Ubuntu 10.10 "Maverick Meerkat" -- gcc-4.4.5
    AddGnuCPlusPlusIncludePaths("/usr/include/c++/4.4",
                                "i686-linux-gnu", "", "64", triple);
    // The rest of 10.10 is the same as previous versions.

    // Ubuntu 10.04 LTS "Lucid Lynx" -- gcc-4.4.3
    // Ubuntu 9.10 "Karmic Koala"    -- gcc-4.4.1
    // Debian 6.0 "squeeze"          -- gcc-4.4.2
    AddGnuCPlusPlusIncludePaths("/usr/include/c++/4.4",
                                "x86_64-linux-gnu", "32", "", triple);
    AddGnuCPlusPlusIncludePaths("/usr/include/c++/4.4",
                                "i486-linux-gnu", "", "64", triple);
    AddGnuCPlusPlusIncludePaths("/usr/include/c++/4.4",
                                "arm-linux-gnueabi", "", "", triple);
    // Ubuntu 9.04 "Jaunty Jackalope" -- gcc-4.3.3
    // Ubuntu 8.10 "Intrepid Ibex"    -- gcc-4.3.2
    // Debian 5.0 "lenny"             -- gcc-4.3.2
    AddGnuCPlusPlusIncludePaths("/usr/include/c++/4.3",
                                "x86_64-linux-gnu", "32", "", triple);
    AddGnuCPlusPlusIncludePaths("/usr/include/c++/4.3",
                                "i486-linux-gnu", "", "64", triple);
    AddGnuCPlusPlusIncludePaths("/usr/include/c++/4.3",
                                "arm-linux-gnueabi", "", "", triple);
    // Ubuntu 8.04.4 LTS "Hardy Heron"     -- gcc-4.2.4
    // Ubuntu 8.04.[0-3] LTS "Hardy Heron" -- gcc-4.2.3
    AddGnuCPlusPlusIncludePaths("/usr/include/c++/4.2",
                                "x86_64-linux-gnu", "32", "", triple);
    AddGnuCPlusPlusIncludePaths("/usr/include/c++/4.2",
                                "i486-linux-gnu", "", "64", triple);
    // Ubuntu 7.10 "Gutsy Gibbon" -- gcc-4.1.3
    AddGnuCPlusPlusIncludePaths("/usr/include/c++/4.1",
                                "x86_64-linux-gnu", "32", "", triple);
    AddGnuCPlusPlusIncludePaths("/usr/include/c++/4.1",
                                "i486-linux-gnu", "", "64", triple);

    //===------------------------------------------------------------------===//
    // Redhat based distros.
    //===------------------------------------------------------------------===//
    // Fedora 15 (GCC 4.6.1)
    AddGnuCPlusPlusIncludePaths("/usr/include/c++/4.6.1",
                                "x86_64-redhat-linux", "32", "", triple);
    AddGnuCPlusPlusIncludePaths("/usr/include/c++/4.6.1",
                                "i686-redhat-linux", "", "", triple);
    // Fedora 15 (GCC 4.6.0)
    AddGnuCPlusPlusIncludePaths("/usr/include/c++/4.6.0",
                                "x86_64-redhat-linux", "32", "", triple);
    AddGnuCPlusPlusIncludePaths("/usr/include/c++/4.6.0",
                                "i686-redhat-linux", "", "", triple);
    // Fedora 14
    AddGnuCPlusPlusIncludePaths("/usr/include/c++/4.5.1",
                                "x86_64-redhat-linux", "32", "", triple);
    AddGnuCPlusPlusIncludePaths("/usr/include/c++/4.5.1",
                                "i686-redhat-linux", "", "", triple);
    // RHEL5(gcc44)
    AddGnuCPlusPlusIncludePaths("/usr/include/c++/4.4.4",
                                "x86_64-redhat-linux6E", "32", "", triple);
    // Fedora 13
    AddGnuCPlusPlusIncludePaths("/usr/include/c++/4.4.4",
                                "x86_64-redhat-linux", "32", "", triple);
    AddGnuCPlusPlusIncludePaths("/usr/include/c++/4.4.4",
                                "i686-redhat-linux","", "", triple);
    // Fedora 12
    AddGnuCPlusPlusIncludePaths("/usr/include/c++/4.4.3",
                                "x86_64-redhat-linux", "32", "", triple);
    AddGnuCPlusPlusIncludePaths("/usr/include/c++/4.4.3",
                                "i686-redhat-linux","", "", triple);
    // Fedora 12 (pre-FEB-2010)
    AddGnuCPlusPlusIncludePaths("/usr/include/c++/4.4.2",
                                "x86_64-redhat-linux", "32", "", triple);
    AddGnuCPlusPlusIncludePaths("/usr/include/c++/4.4.2",
                                "i686-redhat-linux","", "", triple);
    // Fedora 11
    AddGnuCPlusPlusIncludePaths("/usr/include/c++/4.4.1",
                                "x86_64-redhat-linux", "32", "", triple);
    AddGnuCPlusPlusIncludePaths("/usr/include/c++/4.4.1",
                                "i586-redhat-linux","", "", triple);
    // Fedora 10
    AddGnuCPlusPlusIncludePaths("/usr/include/c++/4.3.2",
                                "x86_64-redhat-linux", "32", "", triple);
    AddGnuCPlusPlusIncludePaths("/usr/include/c++/4.3.2",
                                "i386-redhat-linux","", "", triple);
    // Fedora 9
    AddGnuCPlusPlusIncludePaths("/usr/include/c++/4.3.0",
                                "x86_64-redhat-linux", "32", "", triple);
    AddGnuCPlusPlusIncludePaths("/usr/include/c++/4.3.0",
                                "i386-redhat-linux", "", "", triple);
    // Fedora 8
    AddGnuCPlusPlusIncludePaths("/usr/include/c++/4.1.2",
                                "x86_64-redhat-linux", "", "", triple);
    AddGnuCPlusPlusIncludePaths("/usr/include/c++/4.1.2",
                                "i386-redhat-linux", "", "", triple);
      
    // RHEL 5
    AddGnuCPlusPlusIncludePaths("/usr/include/c++/4.1.1",
                                "x86_64-redhat-linux", "32", "", triple);
    AddGnuCPlusPlusIncludePaths("/usr/include/c++/4.1.1",
                                "i386-redhat-linux", "", "", triple);


    //===------------------------------------------------------------------===//

    // Exherbo (2010-01-25)
    AddGnuCPlusPlusIncludePaths("/usr/include/c++/4.4.3",
                                "x86_64-pc-linux-gnu", "32", "", triple);
    AddGnuCPlusPlusIncludePaths("/usr/include/c++/4.4.3",
                                "i686-pc-linux-gnu", "", "", triple);

    // openSUSE 11.1 32 bit
    AddGnuCPlusPlusIncludePaths("/usr/include/c++/4.3",
                                "i586-suse-linux", "", "", triple);
    // openSUSE 11.1 64 bit
    AddGnuCPlusPlusIncludePaths("/usr/include/c++/4.3",
                                "x86_64-suse-linux", "32", "", triple);
    // openSUSE 11.2
    AddGnuCPlusPlusIncludePaths("/usr/include/c++/4.4",
                                "i586-suse-linux", "", "", triple);
    AddGnuCPlusPlusIncludePaths("/usr/include/c++/4.4",
                                "x86_64-suse-linux", "", "", triple);

    // openSUSE 11.4
    AddGnuCPlusPlusIncludePaths("/usr/include/c++/4.5",
                                "i586-suse-linux", "", "", triple);
    AddGnuCPlusPlusIncludePaths("/usr/include/c++/4.5",
                                "x86_64-suse-linux", "", "", triple);

    // openSUSE 12.1
    AddGnuCPlusPlusIncludePaths("/usr/include/c++/4.6",
                                "i586-suse-linux", "", "", triple);
    AddGnuCPlusPlusIncludePaths("/usr/include/c++/4.6",
                                "x86_64-suse-linux", "", "", triple);
    // Arch Linux 2008-06-24
    AddGnuCPlusPlusIncludePaths("/usr/include/c++/4.3.1",
                                "i686-pc-linux-gnu", "", "", triple);
    AddGnuCPlusPlusIncludePaths("/usr/include/c++/4.3.1",
                                "x86_64-unknown-linux-gnu", "", "", triple);

    // Arch Linux gcc 4.6
    AddGnuCPlusPlusIncludePaths("/usr/include/c++/4.6.1",
                                "i686-pc-linux-gnu", "", "", triple);
    AddGnuCPlusPlusIncludePaths("/usr/include/c++/4.6.1",
                                "x86_64-unknown-linux-gnu", "", "", triple);
    AddGnuCPlusPlusIncludePaths("/usr/include/c++/4.6.0",
                                "i686-pc-linux-gnu", "", "", triple);
    AddGnuCPlusPlusIncludePaths("/usr/include/c++/4.6.0",
                                "x86_64-unknown-linux-gnu", "", "", triple);

    // Slackware gcc 4.5.2 (13.37)
    AddGnuCPlusPlusIncludePaths("/usr/include/c++/4.5.2",
                                "i486-slackware-linux", "", "", triple);
    AddGnuCPlusPlusIncludePaths("/usr/include/c++/4.5.2",
                                "x86_64-slackware-linux", "", "", triple);
    // Slackware gcc 4.5.3 (-current)
    AddGnuCPlusPlusIncludePaths("/usr/include/c++/4.5.3",
                                "i486-slackware-linux", "", "", triple);
    AddGnuCPlusPlusIncludePaths("/usr/include/c++/4.5.3",
                                "x86_64-slackware-linux", "", "", triple);

    // Gentoo x86 gcc 4.5.2
    AddGnuCPlusPlusIncludePaths(
      "/usr/lib/gcc/i686-pc-linux-gnu/4.5.2/include/g++-v4",
      "i686-pc-linux-gnu", "", "", triple);
    // Gentoo x86 gcc 4.4.5
    AddGnuCPlusPlusIncludePaths(
      "/usr/lib/gcc/i686-pc-linux-gnu/4.4.5/include/g++-v4",
      "i686-pc-linux-gnu", "", "", triple);
    // Gentoo x86 gcc 4.4.4
    AddGnuCPlusPlusIncludePaths(
      "/usr/lib/gcc/i686-pc-linux-gnu/4.4.4/include/g++-v4",
      "i686-pc-linux-gnu", "", "", triple);
   // Gentoo x86 2010.0 stable
    AddGnuCPlusPlusIncludePaths(
      "/usr/lib/gcc/i686-pc-linux-gnu/4.4.3/include/g++-v4",
      "i686-pc-linux-gnu", "", "", triple);
    // Gentoo x86 2009.1 stable
    AddGnuCPlusPlusIncludePaths(
      "/usr/lib/gcc/i686-pc-linux-gnu/4.3.4/include/g++-v4",
      "i686-pc-linux-gnu", "", "", triple);
    // Gentoo x86 2009.0 stable
    AddGnuCPlusPlusIncludePaths(
      "/usr/lib/gcc/i686-pc-linux-gnu/4.3.2/include/g++-v4",
      "i686-pc-linux-gnu", "", "", triple);
    // Gentoo x86 2008.0 stable
    AddGnuCPlusPlusIncludePaths(
      "/usr/lib/gcc/i686-pc-linux-gnu/4.1.2/include/g++-v4",
      "i686-pc-linux-gnu", "", "", triple);
    // Gentoo x86 llvm-gcc trunk
    AddGnuCPlusPlusIncludePaths(
        "/usr/lib/llvm-gcc-4.2-9999/include/c++/4.2.1",
        "i686-pc-linux-gnu", "", "", triple);

    // Gentoo amd64 gcc 4.5.2
    AddGnuCPlusPlusIncludePaths(
        "/usr/lib/gcc/x86_64-pc-linux-gnu/4.5.2/include/g++-v4",
        "x86_64-pc-linux-gnu", "32", "", triple);
    // Gentoo amd64 gcc 4.4.5
    AddGnuCPlusPlusIncludePaths(
        "/usr/lib/gcc/x86_64-pc-linux-gnu/4.4.5/include/g++-v4",
        "x86_64-pc-linux-gnu", "32", "", triple);
    // Gentoo amd64 gcc 4.4.4
    AddGnuCPlusPlusIncludePaths(
        "/usr/lib/gcc/x86_64-pc-linux-gnu/4.4.4/include/g++-v4",
        "x86_64-pc-linux-gnu", "32", "", triple);
    // Gentoo amd64 gcc 4.4.3
    AddGnuCPlusPlusIncludePaths(
        "/usr/lib/gcc/x86_64-pc-linux-gnu/4.4.3/include/g++-v4",
        "x86_64-pc-linux-gnu", "32", "", triple);
    // Gentoo amd64 gcc 4.3.4
    AddGnuCPlusPlusIncludePaths(
        "/usr/lib/gcc/x86_64-pc-linux-gnu/4.3.4/include/g++-v4",
        "x86_64-pc-linux-gnu", "", "", triple);
    // Gentoo amd64 gcc 4.3.2
    AddGnuCPlusPlusIncludePaths(
        "/usr/lib/gcc/x86_64-pc-linux-gnu/4.3.2/include/g++-v4",
        "x86_64-pc-linux-gnu", "", "", triple);
    // Gentoo amd64 stable
    AddGnuCPlusPlusIncludePaths(
        "/usr/lib/gcc/x86_64-pc-linux-gnu/4.1.2/include/g++-v4",
        "x86_64-pc-linux-gnu", "", "", triple);

    // Gentoo amd64 llvm-gcc trunk
    AddGnuCPlusPlusIncludePaths(
        "/usr/lib/llvm-gcc-4.2-9999/include/c++/4.2.1",
        "x86_64-pc-linux-gnu", "", "", triple);

    break;
  case llvm::Triple::FreeBSD:
    // FreeBSD 8.0
    // FreeBSD 7.3
    AddGnuCPlusPlusIncludePaths("/usr/include/c++/4.2", "", "", "", triple);
    break;
  case llvm::Triple::NetBSD:
    AddGnuCPlusPlusIncludePaths("/usr/include/g++", "", "", "", triple);
    break;
  case llvm::Triple::OpenBSD: {
    std::string t = triple.getTriple();
    if (t.substr(0, 6) == "x86_64")
      t.replace(0, 6, "amd64");
    AddGnuCPlusPlusIncludePaths("/usr/include/g++",
                                t, "", "", triple);
    break;
  }
  case llvm::Triple::Minix:
    AddGnuCPlusPlusIncludePaths("/usr/gnu/include/c++/4.4.3",
                                "", "", "", triple);
    break;
  case llvm::Triple::Solaris:
    // Solaris - Fall though..
  case llvm::Triple::AuroraUX:
    // AuroraUX
    AddGnuCPlusPlusIncludePaths("/opt/gcc4/include/c++/4.2.4",
                                "i386-pc-solaris2.11", "", "", triple);
    break;
  default:
    break;
  }
}

void InitHeaderSearch::AddDefaultIncludePaths(const LangOptions &Lang,
                                              const llvm::Triple &triple,
                                            const HeaderSearchOptions &HSOpts) {
  if (Lang.CPlusPlus && HSOpts.UseStandardCXXIncludes &&
      HSOpts.UseStandardSystemIncludes) {
    if (HSOpts.UseLibcxx) {
      if (triple.isOSDarwin()) {
        // On Darwin, libc++ may be installed alongside the compiler in
        // lib/c++/v1.
        llvm::sys::Path P(HSOpts.ResourceDir);
        if (!P.isEmpty()) {
          P.eraseComponent();  // Remove version from foo/lib/clang/version
          P.eraseComponent();  // Remove clang from foo/lib/clang
          
          // Get foo/lib/c++/v1
          P.appendComponent("c++");
          P.appendComponent("v1");
          AddPath(P.str(), CXXSystem, true, false, false, true);
        }
      }
      
      AddPath("/usr/include/c++/v1", CXXSystem, true, false, false);
    } else {
      AddDefaultCPlusPlusIncludePaths(triple, HSOpts);
    }
  }

  AddDefaultCIncludePaths(triple, HSOpts);

  // Add the default framework include paths on Darwin.
  if (HSOpts.UseStandardSystemIncludes) {
    if (triple.isOSDarwin()) {
      AddPath("/System/Library/Frameworks", System, true, false, true);
      AddPath("/Library/Frameworks", System, true, false, true);
    }
  }
}

/// RemoveDuplicates - If there are duplicate directory entries in the specified
/// search list, remove the later (dead) ones.  Returns the number of non-system
/// headers removed, which is used to update NumAngled.
static unsigned RemoveDuplicates(std::vector<DirectoryLookup> &SearchList,
                                 unsigned First, bool Verbose) {
  llvm::SmallPtrSet<const DirectoryEntry *, 8> SeenDirs;
  llvm::SmallPtrSet<const DirectoryEntry *, 8> SeenFrameworkDirs;
  llvm::SmallPtrSet<const HeaderMap *, 8> SeenHeaderMaps;
  unsigned NonSystemRemoved = 0;
  for (unsigned i = First; i != SearchList.size(); ++i) {
    unsigned DirToRemove = i;

    const DirectoryLookup &CurEntry = SearchList[i];

    if (CurEntry.isNormalDir()) {
      // If this isn't the first time we've seen this dir, remove it.
      if (SeenDirs.insert(CurEntry.getDir()))
        continue;
    } else if (CurEntry.isFramework()) {
      // If this isn't the first time we've seen this framework dir, remove it.
      if (SeenFrameworkDirs.insert(CurEntry.getFrameworkDir()))
        continue;
    } else {
      assert(CurEntry.isHeaderMap() && "Not a headermap or normal dir?");
      // If this isn't the first time we've seen this headermap, remove it.
      if (SeenHeaderMaps.insert(CurEntry.getHeaderMap()))
        continue;
    }

    // If we have a normal #include dir/framework/headermap that is shadowed
    // later in the chain by a system include location, we actually want to
    // ignore the user's request and drop the user dir... keeping the system
    // dir.  This is weird, but required to emulate GCC's search path correctly.
    //
    // Since dupes of system dirs are rare, just rescan to find the original
    // that we're nuking instead of using a DenseMap.
    if (CurEntry.getDirCharacteristic() != SrcMgr::C_User) {
      // Find the dir that this is the same of.
      unsigned FirstDir;
      for (FirstDir = 0; ; ++FirstDir) {
        assert(FirstDir != i && "Didn't find dupe?");

        const DirectoryLookup &SearchEntry = SearchList[FirstDir];

        // If these are different lookup types, then they can't be the dupe.
        if (SearchEntry.getLookupType() != CurEntry.getLookupType())
          continue;

        bool isSame;
        if (CurEntry.isNormalDir())
          isSame = SearchEntry.getDir() == CurEntry.getDir();
        else if (CurEntry.isFramework())
          isSame = SearchEntry.getFrameworkDir() == CurEntry.getFrameworkDir();
        else {
          assert(CurEntry.isHeaderMap() && "Not a headermap or normal dir?");
          isSame = SearchEntry.getHeaderMap() == CurEntry.getHeaderMap();
        }

        if (isSame)
          break;
      }

      // If the first dir in the search path is a non-system dir, zap it
      // instead of the system one.
      if (SearchList[FirstDir].getDirCharacteristic() == SrcMgr::C_User)
        DirToRemove = FirstDir;
    }

    if (Verbose) {
      llvm::errs() << "ignoring duplicate directory \""
                   << CurEntry.getName() << "\"\n";
      if (DirToRemove != i)
        llvm::errs() << "  as it is a non-system directory that duplicates "
                     << "a system directory\n";
    }
    if (DirToRemove != i)
      ++NonSystemRemoved;

    // This is reached if the current entry is a duplicate.  Remove the
    // DirToRemove (usually the current dir).
    SearchList.erase(SearchList.begin()+DirToRemove);
    --i;
  }
  return NonSystemRemoved;
}


void InitHeaderSearch::Realize(const LangOptions &Lang) {
  // Concatenate ANGLE+SYSTEM+AFTER chains together into SearchList.
  std::vector<DirectoryLookup> SearchList;
  SearchList.reserve(IncludePath.size());

  // Quoted arguments go first.
  for (path_iterator it = IncludePath.begin(), ie = IncludePath.end();
       it != ie; ++it) {
    if (it->first == Quoted)
      SearchList.push_back(it->second);
  }
  // Deduplicate and remember index.
  RemoveDuplicates(SearchList, 0, Verbose);
  unsigned NumQuoted = SearchList.size();

  for (path_iterator it = IncludePath.begin(), ie = IncludePath.end();
       it != ie; ++it) {
    if (it->first == Angled || it->first == IndexHeaderMap)
      SearchList.push_back(it->second);
  }

  RemoveDuplicates(SearchList, NumQuoted, Verbose);
  unsigned NumAngled = SearchList.size();

  for (path_iterator it = IncludePath.begin(), ie = IncludePath.end();
       it != ie; ++it) {
    if (it->first == System ||
        (!Lang.ObjC1 && !Lang.CPlusPlus && it->first == CSystem)    ||
        (/*FIXME !Lang.ObjC1 && */Lang.CPlusPlus  && it->first == CXXSystem)  ||
        (Lang.ObjC1  && !Lang.CPlusPlus && it->first == ObjCSystem) ||
        (Lang.ObjC1  && Lang.CPlusPlus  && it->first == ObjCXXSystem))
      SearchList.push_back(it->second);
  }

  for (path_iterator it = IncludePath.begin(), ie = IncludePath.end();
       it != ie; ++it) {
    if (it->first == After)
      SearchList.push_back(it->second);
  }

  // Remove duplicates across both the Angled and System directories.  GCC does
  // this and failing to remove duplicates across these two groups breaks
  // #include_next.
  unsigned NonSystemRemoved = RemoveDuplicates(SearchList, NumQuoted, Verbose);
  NumAngled -= NonSystemRemoved;

  bool DontSearchCurDir = false;  // TODO: set to true if -I- is set?
  Headers.SetSearchPaths(SearchList, NumQuoted, NumAngled, DontSearchCurDir);

  // If verbose, print the list of directories that will be searched.
  if (Verbose) {
    llvm::errs() << "#include \"...\" search starts here:\n";
    for (unsigned i = 0, e = SearchList.size(); i != e; ++i) {
      if (i == NumQuoted)
        llvm::errs() << "#include <...> search starts here:\n";
      const char *Name = SearchList[i].getName();
      const char *Suffix;
      if (SearchList[i].isNormalDir())
        Suffix = "";
      else if (SearchList[i].isFramework())
        Suffix = " (framework directory)";
      else {
        assert(SearchList[i].isHeaderMap() && "Unknown DirectoryLookup");
        Suffix = " (headermap)";
      }
      llvm::errs() << " " << Name << Suffix << "\n";
    }
    llvm::errs() << "End of search list.\n";
  }
}

void clang::ApplyHeaderSearchOptions(HeaderSearch &HS,
                                     const HeaderSearchOptions &HSOpts,
                                     const LangOptions &Lang,
                                     const llvm::Triple &Triple) {
  InitHeaderSearch Init(HS, HSOpts.Verbose, HSOpts.Sysroot);

  // Add the user defined entries.
  for (unsigned i = 0, e = HSOpts.UserEntries.size(); i != e; ++i) {
    const HeaderSearchOptions::Entry &E = HSOpts.UserEntries[i];
    Init.AddPath(E.Path, E.Group, false, E.IsUserSupplied, E.IsFramework,
                 E.IgnoreSysRoot);
  }

  Init.AddDefaultIncludePaths(Lang, Triple, HSOpts);

  Init.Realize(Lang);
}
