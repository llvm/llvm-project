//===- comgr-logger.h - Global Comgr logging facility ---------------------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Declares COMGR::Logger, a process-global, thread-safe logging facility
/// shared by every Comgr API. APIs emit diagnostics via Logger::emit at a
/// configurable severity (AMD_COMGR_LOG_LEVEL; see
/// COMGR::env::resolveLogLevel()).
///
/// Output goes to two independent destinations:
///   - The global "sink": resolved once from AMD_COMGR_REDIRECT_LOGS (stdout,
///     stderr, or an appended file).
///   - An optional per-thread "capture" stream: installed via LogCaptureScope
///     so messages are also collected into the AMD_COMGR_DATA_KIND_LOG
///     ("comgr.log") data object returned to the caller.
///
/// All writes are mutex-guarded, so concurrent callers share the sink safely.
///
//===----------------------------------------------------------------------===//

#ifndef COMGR_LOGGER_H
#define COMGR_LOGGER_H

#include "comgr-env.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/raw_ostream.h"

#include <memory>
#include <mutex>
#include <string>

namespace COMGR {

/// The severity type shared with the logger; defined in comgr-env alongside the
/// AMD_COMGR_LOG_LEVEL parsing that produces it.
using env::LogLevel;

/// Process-global, thread-safe logging facility. Obtain the shared instance
/// through getLogger(); do not construct directly except for tests.
class Logger {
public:
  /// Construct from the environment (AMD_COMGR_LOG_LEVEL and
  /// AMD_COMGR_REDIRECT_LOGS). Used for the process-global instance.
  Logger();

  /// Construct with an explicit level and non-owning sink (may be null). For
  /// tests.
  Logger(LogLevel Level, llvm::raw_ostream *Sink);

  /// Construct with an explicit level, resolving the sink from @p
  /// RedirectTarget like AMD_COMGR_REDIRECT_LOGS but bypassing the env cache.
  /// Exposed for testing.
  Logger(LogLevel Level, llvm::StringRef RedirectTarget);

  Logger(const Logger &) = delete;
  Logger &operator=(const Logger &) = delete;

  /// Whether @p Severity would be emitted at the current level (None is never
  /// emitted). Use to guard building expensive messages.
  bool isEnabled(LogLevel Severity) const {
    return Severity != LogLevel::None && Level != LogLevel::None &&
           Severity <= Level;
  }

  /// The configured maximum severity that will be emitted.
  LogLevel getLevel() const { return Level; }

  /// Whether a redirect sink (AMD_COMGR_REDIRECT_LOGS) is active. The stream is
  /// not exposed; use writeToSink()/sinkFlush() to stay under the mutex.
  bool hasSink() const { return Sink != nullptr; }

  /// Diagnostic for why the redirect sink could not be opened, empty otherwise.
  /// The action layer surfaces this into comgr.log when hasSink() is false.
  llvm::StringRef getSinkError() const { return SinkError; }

  /// Filename the redirect sink was opened on, empty for a stream sink or when
  /// not redirected. Lets callers reuse the destination without re-classifying.
  llvm::StringRef getRedirectFilename() const { return SinkFilename; }

  /// Write @p Data verbatim to the sink under the mutex (no
  /// prefix/newline/flush) so teed output does not race emit(). No-op when
  /// there is no sink.
  void writeToSink(llvm::StringRef Data);

  /// Flush the global sink under the logger's mutex. No-op when there is no
  /// sink.
  void sinkFlush();

  /// Emit @p Message at @p Severity, prefixed and newline-terminated, to the
  /// sink and the calling thread's capture stream (if any). Thread-safe.
  void emit(LogLevel Severity, const llvm::Twine &Message);

private:
  // Resolve and install the redirect sink from @p RedirectLog (stream for
  // stdout/stderr/"-", else append-mode file; records SinkError on failure).
  void openSink(llvm::StringRef RedirectLog);

  LogLevel Level;

  // The global sink, resolved once at construction. Null when logs are not
  // redirected (AMD_COMGR_REDIRECT_LOGS unset). For a file, owned by SinkFile.
  llvm::raw_ostream *Sink;
  std::unique_ptr<llvm::raw_fd_ostream> SinkFile;

  // Filename the sink was opened on for a file redirect, empty otherwise.
  // Surfaced via getRedirectFilename().
  std::string SinkFilename;

  // Diagnostic recorded when the redirect file could not be opened, empty
  // otherwise. Surfaced via getSinkError().
  std::string SinkError;

  // Guards all writes to Sink and to the active capture stream.
  std::mutex Mutex;
};

/// Return the process-global Logger instance.
Logger &getLogger();

/// Install a capture stream for the current thread for this scope's duration.
/// While active, every Logger::emit on this thread also writes into @p OS, in
/// addition to the global sink. Nesting is supported: the previous capture
/// stream (if any) is restored on destruction.
class LogCaptureScope {
public:
  explicit LogCaptureScope(llvm::raw_ostream &OS);
  ~LogCaptureScope();

  LogCaptureScope(const LogCaptureScope &) = delete;
  LogCaptureScope &operator=(const LogCaptureScope &) = delete;

private:
  llvm::raw_ostream *Previous;
};

/// Return the capture stream installed on the calling thread, or null. Exposed
/// for Logger::emit; callers should use LogCaptureScope to manage it.
llvm::raw_ostream *getThreadCaptureStream();

} // namespace COMGR

#endif // COMGR_LOGGER_H
