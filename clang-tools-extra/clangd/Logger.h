//===--- Logger.h - Logger interface for clangd ------------------*- C++-*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANGD_LOGGER_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANGD_LOGGER_H

#include "llvm/ADT/Twine.h"
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

  enum Level { Debug, Verbose, Info, Error };
  static char indicator(Level L) { return "DVIE"[L]; }

  /// Implementations of this method must be thread-safe.
  virtual void log(Level, const llvm::formatv_object_base &Message) = 0;
};

namespace detail {
const char *debugType(const char *Filename);
void log(Logger::Level, const llvm::formatv_object_base &);

// We often want to consume llvm::Errors by value when passing them to log().
// We automatically wrap them in llvm::fmt_consume() as formatv requires.
template <typename T> T &&wrap(T &&V) { return std::forward<T>(V); }
inline decltype(fmt_consume(llvm::Error::success())) wrap(llvm::Error &&V) {
  return fmt_consume(std::move(V));
}
template <typename... Ts>
void log(Logger::Level L, const char *Fmt, Ts &&... Vals) {
  detail::log(L, llvm::formatv(Fmt, detail::wrap(std::forward<Ts>(Vals))...));
}
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
  void log(Level, const llvm::formatv_object_base &Message) override;

private:
  Logger::Level MinLevel;
  llvm::raw_ostream &Logs;

  std::mutex StreamMutex;
};

} // namespace clangd
} // namespace clang

#endif
