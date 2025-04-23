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

typedef struct {
  char padding[396];
  char gcnArchName[256];
  char padding2[1024];
} hipDeviceProp_t;

typedef enum {
  hipSuccess = 0,
} hipError_t;

typedef hipError_t (*hipGetDeviceCount_t)(int *);
typedef hipError_t (*hipDeviceGet_t)(int *, int);
typedef hipError_t (*hipGetDeviceProperties_t)(hipDeviceProp_t *, int);

extern cl::opt<bool> Verbose;

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
    size_t Pos = S.find_last_of('_');
    StringRef VerStr = (Pos == StringRef::npos) ? S : S.substr(Pos + 1);
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
    if (!DLLNames.empty())
      break;
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
    llvm::errs() << "Failed to load " << DynamicHIPPath << ": " << ErrMsg
                 << '\n';
    return 1;
  }

#define DYNAMIC_INIT_HIP(SYMBOL)                                               \
  {                                                                            \
    void *SymbolPtr = DynlibHandle->getAddressOfSymbol(#SYMBOL);               \
    if (!SymbolPtr) {                                                          \
      llvm::errs() << "Failed to find symbol " << #SYMBOL << '\n';             \
      return 1;                                                                \
    }                                                                          \
    SYMBOL = reinterpret_cast<decltype(SYMBOL)>(SymbolPtr);                    \
  }

  hipGetDeviceCount_t hipGetDeviceCount;
  hipDeviceGet_t hipDeviceGet;
  hipGetDeviceProperties_t hipGetDeviceProperties;

  DYNAMIC_INIT_HIP(hipGetDeviceCount);
  DYNAMIC_INIT_HIP(hipDeviceGet);
  DYNAMIC_INIT_HIP(hipGetDeviceProperties);

#undef DYNAMIC_INIT_HIP

  int deviceCount;
  hipError_t err = hipGetDeviceCount(&deviceCount);
  if (err != hipSuccess) {
    llvm::errs() << "Failed to get device count\n";
    return 1;
  }

  for (int i = 0; i < deviceCount; ++i) {
    int deviceId;
    err = hipDeviceGet(&deviceId, i);
    if (err != hipSuccess) {
      llvm::errs() << "Failed to get device id for ordinal " << i << '\n';
      return 1;
    }

    hipDeviceProp_t prop;
    err = hipGetDeviceProperties(&prop, deviceId);
    if (err != hipSuccess) {
      llvm::errs() << "Failed to get device properties for device " << deviceId
                   << '\n';
      return 1;
    }
    llvm::outs() << prop.gcnArchName << '\n';
  }

  return 0;
}
