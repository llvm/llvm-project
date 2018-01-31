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

namespace clang {
namespace clangd {

/// Main logging function.
/// Logs messages to a global logger, which can be set up by LoggingSesssion.
/// If no logger is registered, writes to llvm::errs().
void log(const llvm::Twine &Message);

/// Interface to allow custom logging in clangd.
class Logger {
public:
  virtual ~Logger() = default;

  /// Implementations of this method must be thread-safe.
  virtual void log(const llvm::Twine &Message) = 0;
};

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

} // namespace clangd
} // namespace clang

#endif
