//===- AMDGPUArchByHIP.cpp - list AMDGPU installed ----------*- C++ -*-----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a tool for detecting name of AMDGPU installed in system
// using HIP runtime. This tool is used by AMDGPU OpenMP and HIP driver.
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Sequence.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ConvertUTF.h"
#include "llvm/Support/DynamicLibrary.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Process.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/VersionTuple.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <string>
#include <vector>

#ifdef _WIN32
#include <windows.h>
#endif

using namespace llvm;

// R0600 struct layout (HIP 6.x+)
typedef struct alignas(8) {
  char padding[1160];
  char gcnArchName[256];
  char padding2[56];
} hipDeviceProp_tR0600;

// R0000 struct layout (legacy)
typedef struct alignas(8) {
  char padding[396];
  char gcnArchName[256];
  char padding2[1024];
} hipDeviceProp_tR0000;

typedef enum {
  hipSuccess = 0,
} hipError_t;

typedef hipError_t (*hipGetDeviceCount_t)(int *);
typedef hipError_t (*hipGetDevicePropertiesR0600_t)(hipDeviceProp_tR0600 *,
                                                    int);
typedef hipError_t (*hipGetDevicePropertiesR0000_t)(hipDeviceProp_tR0000 *,
                                                    int);
typedef hipError_t (*hipGetDeviceProperties_t)(hipDeviceProp_tR0000 *, int);
typedef hipError_t (*hipRuntimeGetVersion_t)(int *);
typedef const char *(*hipGetErrorString_t)(hipError_t);

extern cl::opt<bool> Verbose;

cl::OptionCategory AMDGPUArchByHIPCategory("amdgpu-arch (HIP) options");

enum class HipApiVersion {
  Auto,       // Automatic fallback (R0600 -> R0000 -> unversioned)
  R0600,      // Force R0600 API (HIP 6.x+)
  R0000,      // Force R0000 API (legacy HIP)
  Unversioned // Force unversioned API (very old HIP)
};

static cl::opt<HipApiVersion> HipApi(
    "hip-api-version", cl::desc("Select HIP API version for device properties"),
    cl::values(clEnumValN(HipApiVersion::Auto, "auto",
                          "Auto-detect (R0600 -> R0000 -> unversioned)"),
               clEnumValN(HipApiVersion::R0600, "r0600", "Force R0600 API"),
               clEnumValN(HipApiVersion::R0000, "r0000", "Force R0000 API"),
               clEnumValN(HipApiVersion::Unversioned, "unversioned",
                          "Force unversioned API")),
    cl::init(HipApiVersion::Auto), cl::cat(AMDGPUArchByHIPCategory));

#ifdef _WIN32
static std::vector<std::string> getSearchPaths() {
  std::vector<std::string> Paths;

  // Get the directory of the current executable
  if (auto MainExe = sys::fs::getMainExecutable(nullptr, nullptr);
      !MainExe.empty())
    Paths.push_back(sys::path::parent_path(MainExe).str());

  // Get the system directory
  wchar_t SystemDirectory[MAX_PATH];
  if (GetSystemDirectoryW(SystemDirectory, MAX_PATH) > 0) {
    std::string Utf8SystemDir;
    if (convertUTF16ToUTF8String(
            ArrayRef<UTF16>(reinterpret_cast<const UTF16 *>(SystemDirectory),
                            wcslen(SystemDirectory)),
            Utf8SystemDir))
      Paths.push_back(Utf8SystemDir);
  }

  // Get the Windows directory
  wchar_t WindowsDirectory[MAX_PATH];
  if (GetWindowsDirectoryW(WindowsDirectory, MAX_PATH) > 0) {
    std::string Utf8WindowsDir;
    if (convertUTF16ToUTF8String(
            ArrayRef<UTF16>(reinterpret_cast<const UTF16 *>(WindowsDirectory),
                            wcslen(WindowsDirectory)),
            Utf8WindowsDir))
      Paths.push_back(Utf8WindowsDir);
  }

  // Get the current working directory
  SmallVector<char, 256> CWD;
  if (sys::fs::current_path(CWD))
    Paths.push_back(std::string(CWD.begin(), CWD.end()));

  // Get the PATH environment variable
  if (std::optional<std::string> PathEnv = sys::Process::GetEnv("PATH")) {
    SmallVector<StringRef, 16> PathList;
    StringRef(*PathEnv).split(PathList, sys::EnvPathSeparator);
    for (auto &Path : PathList)
      Paths.push_back(Path.str());
  }

  return Paths;
}

// Custom comparison function for dll name
static bool compareVersions(StringRef A, StringRef B) {
  auto ParseVersion = [](StringRef S) -> VersionTuple {
    StringRef Filename = sys::path::filename(S);
    size_t Pos = Filename.find_last_of('_');
    if (Pos == StringRef::npos)
      return VersionTuple();

    StringRef VerStr = Filename.substr(Pos + 1);
    size_t DotPos = VerStr.find('.');
    if (DotPos != StringRef::npos)
      VerStr = VerStr.substr(0, DotPos);

    VersionTuple Vt;
    (void)Vt.tryParse(VerStr);
    return Vt;
  };

  VersionTuple VtA = ParseVersion(A);
  VersionTuple VtB = ParseVersion(B);
  return VtA > VtB;
}
#endif

// On Windows, prefer amdhip64_n.dll where n is ROCm major version and greater
// value of n takes precedence. If amdhip64_n.dll is not found, fall back to
// amdhip64.dll. The reason is that a normal driver installation only has
// amdhip64_n.dll but we do not know what n is since this program may be used
// with a future version of HIP runtime.
//
// On Linux, always use default libamdhip64.so.
static std::pair<std::string, bool> findNewestHIPDLL() {
#ifdef _WIN32
  StringRef HipDLLPrefix = "amdhip64_";
  StringRef HipDLLSuffix = ".dll";

  std::vector<std::string> SearchPaths = getSearchPaths();
  std::vector<std::string> DLLNames;

  for (const auto &Dir : SearchPaths) {
    std::error_code EC;
    for (sys::fs::directory_iterator DirIt(Dir, EC), DirEnd;
         DirIt != DirEnd && !EC; DirIt.increment(EC)) {
      StringRef Filename = sys::path::filename(DirIt->path());
      if (Filename.starts_with(HipDLLPrefix) &&
          Filename.ends_with(HipDLLSuffix))
        DLLNames.push_back(sys::path::convert_to_slash(DirIt->path()));
    }
  }

  if (DLLNames.empty())
    return {"amdhip64.dll", true};

  llvm::sort(DLLNames, compareVersions);
  return {DLLNames[0], false};
#else
  // On Linux, fallback to default shared object
  return {"libamdhip64.so", true};
#endif
}

int printGPUsByHIP() {
  auto [DynamicHIPPath, IsFallback] = findNewestHIPDLL();

  if (Verbose) {
    if (IsFallback)
      outs() << "Using default HIP runtime: " << DynamicHIPPath << '\n';
    else
      outs() << "Found HIP runtime: " << DynamicHIPPath << '\n';
  }

  std::string ErrMsg;
  auto DynlibHandle = std::make_unique<llvm::sys::DynamicLibrary>(
      llvm::sys::DynamicLibrary::getPermanentLibrary(DynamicHIPPath.c_str(),
                                                     &ErrMsg));
  if (!DynlibHandle->isValid()) {
    if (Verbose)
      llvm::errs() << "Failed to load " << DynamicHIPPath << ": " << ErrMsg
                   << '\n';
    return 1;
  }

  if (Verbose)
    outs() << "Successfully loaded HIP runtime library\n";

#define DYNAMIC_INIT_HIP(SYMBOL)                                               \
  {                                                                            \
    void *SymbolPtr = DynlibHandle->getAddressOfSymbol(#SYMBOL);               \
    if (!SymbolPtr) {                                                          \
      llvm::errs() << "Failed to find symbol " << #SYMBOL << '\n';             \
      return 1;                                                                \
    }                                                                          \
    if (Verbose)                                                               \
      outs() << "Found symbol: " << #SYMBOL << '\n';                           \
    SYMBOL = reinterpret_cast<decltype(SYMBOL)>(SymbolPtr);                    \
  }

  hipGetDeviceCount_t hipGetDeviceCount;
  hipRuntimeGetVersion_t hipRuntimeGetVersion = nullptr;
  hipGetDevicePropertiesR0600_t hipGetDevicePropertiesR0600 = nullptr;
  hipGetDevicePropertiesR0000_t hipGetDevicePropertiesR0000 = nullptr;
  hipGetDeviceProperties_t hipGetDeviceProperties = nullptr;
  hipGetErrorString_t hipGetErrorString = nullptr;

  DYNAMIC_INIT_HIP(hipGetDeviceCount);

#undef DYNAMIC_INIT_HIP

  auto LoadSymbol = [&](const char *Name, auto &FuncPtr,
                        const char *Desc = "") {
    void *Sym = DynlibHandle->getAddressOfSymbol(Name);
    if (Sym) {
      FuncPtr = reinterpret_cast<decltype(FuncPtr)>(Sym);
      if (Verbose)
        outs() << "Found symbol: " << Name << (Desc[0] ? " " : "") << Desc
               << '\n';
      return true;
    }
    return false;
  };

  LoadSymbol("hipGetErrorString", hipGetErrorString);

  if (LoadSymbol("hipRuntimeGetVersion", hipRuntimeGetVersion)) {
    int RuntimeVersion = 0;
    if (hipRuntimeGetVersion(&RuntimeVersion) == hipSuccess) {
      int Major = RuntimeVersion / 10000000;
      int Minor = (RuntimeVersion / 100000) % 100;
      int Patch = RuntimeVersion % 100000;
      if (Verbose)
        outs() << "HIP Runtime Version: " << Major << "." << Minor << "."
               << Patch << '\n';
    }
  }

  LoadSymbol("hipGetDevicePropertiesR0600", hipGetDevicePropertiesR0600,
             "(HIP 6.x+ API)");
  LoadSymbol("hipGetDevicePropertiesR0000", hipGetDevicePropertiesR0000,
             "(legacy API)");
  if (!hipGetDevicePropertiesR0600 && !hipGetDevicePropertiesR0000)
    LoadSymbol("hipGetDeviceProperties", hipGetDeviceProperties,
               "(unversioned legacy API)");

  int DeviceCount;
  if (Verbose)
    outs() << "Calling hipGetDeviceCount...\n";
  hipError_t Err = hipGetDeviceCount(&DeviceCount);
  if (Err != hipSuccess) {
    llvm::errs() << "Failed to get device count";
    if (hipGetErrorString) {
      llvm::errs() << ": " << hipGetErrorString(Err);
    }
    llvm::errs() << " (error code: " << Err << ")\n";
    return 1;
  }

  if (Verbose)
    outs() << "Found " << DeviceCount << " device(s)\n";

  auto TryGetProperties = [&](auto *ApiFunc, auto *DummyProp, const char *Name,
                              int DeviceId) -> std::string {
    if (!ApiFunc)
      return "";

    if (Verbose)
      outs() << "Using " << Name << "...\n";

    using PropType = std::remove_pointer_t<decltype(DummyProp)>;
    PropType Prop;
    hipError_t Err = ApiFunc(&Prop, DeviceId);

    if (Err == hipSuccess) {
      if (Verbose) {
        outs() << Name << " struct: sizeof = " << sizeof(PropType)
               << " bytes, offsetof(gcnArchName) = "
               << offsetof(PropType, gcnArchName) << " bytes\n";
      }
      return Prop.gcnArchName;
    }

    if (Verbose)
      llvm::errs() << Name << " failed (error code: " << Err << ")\n";
    return "";
  };

  for (auto I : llvm::seq(DeviceCount)) {
    if (Verbose)
      outs() << "Processing device " << I << "...\n";

    std::string ArchName;
    auto TryR0600 = [&](int Dev) -> bool {
      if (!hipGetDevicePropertiesR0600)
        return false;
      ArchName = TryGetProperties(hipGetDevicePropertiesR0600,
                                  (hipDeviceProp_tR0600 *)nullptr,
                                  "R0600 API (HIP 6.x+)", Dev);
      return !ArchName.empty();
    };
    auto TryR0000 = [&](int Dev) -> bool {
      if (!hipGetDevicePropertiesR0000)
        return false;
      ArchName = TryGetProperties(hipGetDevicePropertiesR0000,
                                  (hipDeviceProp_tR0000 *)nullptr,
                                  "R0000 API (legacy HIP)", Dev);
      return !ArchName.empty();
    };
    auto TryUnversioned = [&](int Dev) -> bool {
      if (!hipGetDeviceProperties)
        return false;
      ArchName = TryGetProperties(hipGetDeviceProperties,
                                  (hipDeviceProp_tR0000 *)nullptr,
                                  "unversioned API (very old HIP)", Dev);
      return !ArchName.empty();
    };

    [[maybe_unused]] bool OK;
    switch (HipApi) {
    case HipApiVersion::Auto:
      OK = TryR0600(I) || TryR0000(I) || TryUnversioned(I);
      break;
    case HipApiVersion::R0600:
      OK = TryR0600(I);
      break;
    case HipApiVersion::R0000:
      OK = TryR0000(I);
      break;
    case HipApiVersion::Unversioned:
      OK = TryUnversioned(I);
    }

    if (ArchName.empty()) {
      llvm::errs() << "Failed to get device properties for device " << I
                   << " - no APIs available or all failed\n";
      return 1;
    }

    if (Verbose)
      outs() << "Device " << I << " arch name: ";
    llvm::outs() << ArchName << '\n';
  }

  return 0;
}
