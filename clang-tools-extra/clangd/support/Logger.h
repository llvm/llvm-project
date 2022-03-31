//===--- Logger.h - Logger interface for clangd ------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANGD_SUPPORT_LOGGER_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANGD_SUPPORT_LOGGER_H

#include "llvm/Support/Debug.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FormatAdapters.h"
#include "llvm/Support/FormatVariadic.h"
#include <mutex>

namespace clang {
namespace clangd {

/// Interface to allow custom logging in clangd.
class Logger {
public:
  virtual ~Logger() = default;

  /// The significance or severity of this message.
  /// Typically used to filter the output to an interesting level.
  enum Level : unsigned char { Debug, Verbose, Info, Error };
  static char indicator(Level L) { return "DVIE"[L]; }

  /// Implementations of this method must be thread-safe.
  virtual void log(Level, const char *Fmt,
                   const llvm::formatv_object_base &Message) = 0;
};

namespace detail {
const char *debugType(const char *Filename);
void logImpl(Logger::Level, const char *Fmt, const llvm::formatv_object_base &);

// We often want to consume llvm::Errors by value when passing them to log().
// We automatically wrap them in llvm::fmt_consume() as formatv requires.
template <typename T> T &&wrap(T &&V) { return std::forward<T>(V); }
inline decltype(fmt_consume(llvm::Error::success())) wrap(llvm::Error &&V) {
  return fmt_consume(std::move(V));
}
template <typename... Ts>
void log(Logger::Level L, const char *Fmt, Ts &&... Vals) {
  detail::logImpl(L, Fmt,
                  llvm::formatv(Fmt, detail::wrap(std::forward<Ts>(Vals))...));
}

llvm::Error error(std::error_code, std::string &&);
} // namespace detail

// Clangd logging functions write to a global logger set by LoggingSession.
// If no logger is registered, writes to llvm::errs().
// All accept llvm::formatv()-style arguments, e.g. log("Text={0}", Text).

// elog() is used for "loud" errors and warnings.
// This level is often visible to users.
template <typename... Ts> void elog(const char *Fmt, Ts &&... Vals) {
  detail::log(Logger::Error, Fmt, std::forward<Ts>(Vals)...);
}
// log() is used for information important to understand a clangd session.
// e.g. the names of LSP messages sent are logged at this level.
// This level could be enabled in production builds to allow later inspection.
template <typename... Ts> void log(const char *Fmt, Ts &&... Vals) {
  detail::log(Logger::Info, Fmt, std::forward<Ts>(Vals)...);
}
// vlog() is used for details often needed for debugging clangd sessions.
// This level would typically be enabled for clangd developers.
template <typename... Ts> void vlog(const char *Fmt, Ts &&... Vals) {
  detail::log(Logger::Verbose, Fmt, std::forward<Ts>(Vals)...);
}
// error() constructs an llvm::Error object, using formatv()-style arguments.
// It is not automatically logged! (This function is a little out of place).
// The error simply embeds the message string.
template <typename... Ts>
llvm::Error error(std::error_code EC, const char *Fmt, Ts &&... Vals) {
  // We must render the formatv_object eagerly, while references are valid.
  return detail::error(
      EC, llvm::formatv(Fmt, detail::wrap(std::forward<Ts>(Vals))...).str());
}
// Overload with no error_code conversion, the error will be inconvertible.
template <typename... Ts> llvm::Error error(const char *Fmt, Ts &&... Vals) {
  return detail::error(
      llvm::inconvertibleErrorCode(),
      llvm::formatv(Fmt, detail::wrap(std::forward<Ts>(Vals))...).str());
}
// Overload to avoid formatv complexity for simple strings.
inline llvm::Error error(std::error_code EC, std::string Msg) {
  return detail::error(EC, std::move(Msg));
}
// Overload for simple strings with no error_code conversion.
inline llvm::Error error(std::string Msg) {
  return detail::error(llvm::inconvertibleErrorCode(), std::move(Msg));
}

// dlog only logs if --debug was passed, or --debug_only=Basename.
// This level would be enabled in a targeted way when debugging.
#define dlog(...)                                                              \
  DEBUG_WITH_TYPE(::clang::clangd::detail::debugType(__FILE__),                \
                  ::clang::clangd::detail::log(Logger::Debug, __VA_ARGS__))

/// Only one LoggingSession can be active at a time.
class LoggingSession {
public:
  LoggingSession(clangd::Logger &Instance);
  ~LoggingSession();

  LoggingSession(LoggingSession &&) = delete;
  LoggingSession &operator=(LoggingSession &&) = delete;

  LoggingSession(LoggingSession const &) = delete;
  LoggingSession &operator=(LoggingSession const &) = delete;
};

// Logs to an output stream, such as stderr.
class StreamLogger : public Logger {
public:
  StreamLogger(llvm::raw_ostream &Logs, Logger::Level MinLevel)
      : MinLevel(MinLevel), Logs(Logs) {}

  /// Write a line to the logging stream.
  void log(Level, const char *Fmt,
           const llvm::formatv_object_base &Message) override;

private:
  Logger::Level MinLevel;
  llvm::raw_ostream &Logs;

  std::mutex StreamMutex;
};

} // namespace clangd
} // namespace clang

#endif
