//===- comgr-env.h - Comgr environment variables --------------------------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef COMGR_ENV_H
#define COMGR_ENV_H

#include "llvm/ADT/StringRef.h"

namespace COMGR {
namespace env {

enum class LogLevel {
  None = 0,
  Error,
  Warning,
  Info,
  Debug,
};

/// Parse @p Requested (AMD_COMGR_LOG_LEVEL, may be empty) into a threshold.
/// When empty or not an integer, returns Debug if @p VerboseFallback is set.
LogLevel parseLogLevel(llvm::StringRef Requested, bool VerboseFallback);

/// Resolve the log level from AMD_COMGR_LOG_LEVEL, using the
/// AMD_COMGR_EMIT_VERBOSE_LOGS back-compat fallback via parseLogLevel().
LogLevel resolveLogLevel();

/// Return whether the environment requests temps be saved.
bool shouldSaveTemps();
bool shouldSaveLLVMTemps();
std::optional<bool> shouldUseVFS();

/// If the environment requests logs be redirected, return the string identifier
/// of where to redirect. Otherwise return @p None.
std::optional<llvm::StringRef> getRedirectLogs();

/// Return whether the environment requests verbose logging.
bool shouldEmitVerboseLogs();

/// Return whether the environment requests time statistics collection.
bool needTimeStatistics();

/// Return granularity (ms, us, ns) units per second
uint32_t getGranularityUnitsPerSecond();

/// Return granularity of time statistics (ms, us, ns)
llvm::StringRef getTimeStatisticsGranularity();

/// If environment variable LLVM_PATH is set, return the environment variable,
/// otherwise return the default LLVM path.
llvm::StringRef getLLVMPath();

/// If environment variable AMD_COMGR_CACHE_POLICY is set, return the
/// environment variable, otherwise return empty
llvm::StringRef getCachePolicy();

/// If environment variable AMD_COMGR_CACHE_DIR is set, return the environment
/// variable, otherwise return the default path: On Linux it's typically
/// $HOME/.cache/comgr_cache (depends on XDG_CACHE_HOME)
llvm::StringRef getCacheDirectory();

/// If environment variable AMD_COMGR_DRIVER_OPTIONS_APPEND is set, return the
/// space-separated options to append to clang driver invocations.
llvm::StringRef getDriverOptionsAppend();

} // namespace env
} // namespace COMGR

#endif // COMGR_ENV_H
