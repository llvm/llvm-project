//===- comgr-env.cpp - Comgr environment variables ------------------------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file implements the management of Comgr's environment variables. See
/// amd/comgr/README.md for descriptions of these.
///
//===----------------------------------------------------------------------===//

#include "comgr-env.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/VirtualFileSystem.h"

#include <cstdlib>

using namespace llvm;

// Use secure_getenv() on glibc so env-controlled paths are ignored under
// AT_SECURE; no such concept elsewhere, so fall back to getenv().
#if defined(__GLIBC__)
#define COMGR_GETENV secure_getenv
#else
#define COMGR_GETENV getenv
#endif

namespace COMGR {
namespace env {

bool shouldSaveTemps() {
  static char *SaveTemps = COMGR_GETENV("AMD_COMGR_SAVE_TEMPS");
  return SaveTemps && StringRef(SaveTemps) != "0";
}

bool shouldSaveLLVMTemps() {
  static char *SaveTemps = COMGR_GETENV("AMD_COMGR_SAVE_LLVM_TEMPS");
  return SaveTemps && StringRef(SaveTemps) != "0";
}

bool shouldAddEntryTrampolineSymbols() {
  // Opt-in (exactly "1"): the B0->B0 fast path skips the debug-only stub
  // symbols by default on the load-time-critical path.
  static char *AddSyms = COMGR_GETENV("AMD_COMGR_HOTSWAP_ENTRY_STUB_SYMBOLS");
  return AddSyms && StringRef(AddSyms) == "1";
}

std::optional<bool> shouldUseVFS() {
  if (shouldSaveTemps())
    return false;

  static char *UseVFS = COMGR_GETENV("AMD_COMGR_USE_VFS");
  if (UseVFS) {
    if (StringRef(UseVFS) == "0")
      return false;
    else if (StringRef(UseVFS) == "1")
      return true;
  }

  return std::nullopt;
}

std::optional<StringRef> getRedirectLogs() {
  static char *RedirectLogs = COMGR_GETENV("AMD_COMGR_REDIRECT_LOGS");
  if (!RedirectLogs || StringRef(RedirectLogs) == "0") {
    return std::nullopt;
  }
  return StringRef(RedirectLogs);
}

bool needTimeStatistics() {
  static char *TimeStatistics = COMGR_GETENV("AMD_COMGR_TIME_STATISTICS");
  return TimeStatistics && StringRef(TimeStatistics) != "0";
}

uint32_t getGranularityUnitsPerSecond() {
  StringRef G = getTimeStatisticsGranularity();
  if (G == "us")
    return 1e6;
  else if (G == "ns")
    return 1e9;
  return 1e3;
}

llvm::StringRef getTimeStatisticsGranularity() {
  static const char *TimeStatisticsGranularity =
      COMGR_GETENV("AMD_COMGR_TIME_STATISTICS_GRANULARITY");
  if (!TimeStatisticsGranularity)
    return "ms";
  StringRef G(TimeStatisticsGranularity);
  if (G == "ms" || G == "us" || G == "ns")
    return G;
  return "ms";
}

bool shouldEmitVerboseLogs() {
  static char *VerboseLogs = COMGR_GETENV("AMD_COMGR_EMIT_VERBOSE_LOGS");
  return VerboseLogs && StringRef(VerboseLogs) != "0";
}

LogLevel parseLogLevel(StringRef Requested, bool VerboseFallback) {
  // Unset or non-integer: default to Debug when verbose logs are requested
  // (back-compat with AMD_COMGR_EMIT_VERBOSE_LOGS), else Error so errors show.
  unsigned Numeric;
  if (Requested.getAsInteger(10, Numeric))
    return VerboseFallback ? LogLevel::Debug : LogLevel::Error;

  unsigned Max = static_cast<unsigned>(LogLevel::Debug);
  return static_cast<LogLevel>(std::min(Numeric, Max));
}

LogLevel resolveLogLevel() {
  static const char *LogThreshold = getenv("AMD_COMGR_LOG_LEVEL");
  StringRef Requested = LogThreshold ? StringRef(LogThreshold) : StringRef();
  return parseLogLevel(Requested, shouldEmitVerboseLogs());
}

llvm::StringRef getLLVMPath() {
  static const char *EnvLLVMPath = COMGR_GETENV("LLVM_PATH");
  return EnvLLVMPath ? EnvLLVMPath : "";
}

StringRef getCachePolicy() {
  static const char *EnvCachePolicy = COMGR_GETENV("AMD_COMGR_CACHE_POLICY");
  return EnvCachePolicy ? EnvCachePolicy : "";
}

StringRef getCacheDirectory() {
  // By default the cache is enabled
  static const char *Enable = COMGR_GETENV("AMD_COMGR_CACHE");
  bool CacheDisabled = StringRef(Enable) == "0";
  if (CacheDisabled)
    return "";

  StringRef EnvCacheDirectory = COMGR_GETENV("AMD_COMGR_CACHE_DIR");
  if (!EnvCacheDirectory.empty())
    return EnvCacheDirectory;

  // mark Result as static to keep it cached across calls
  static SmallString<256> Result;
  if (!Result.empty())
    return Result;

  if (sys::path::cache_directory(Result)) {
    sys::path::append(Result, "comgr");
    return Result;
  }

  return "";
}

StringRef getDriverOptionsAppend() {
  static const char *Options = COMGR_GETENV("AMD_COMGR_DRIVER_OPTIONS_APPEND");
  return Options ? Options : "";
}

} // namespace env
} // namespace COMGR
