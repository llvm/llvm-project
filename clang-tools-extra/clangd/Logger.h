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

/// Interface to allow custom logging in clangd.
class Logger {
public:
  virtual ~Logger() = default;

  /// Implementations of this method must be thread-safe.
  virtual void log(const llvm::Twine &Message) = 0;
};

/// Logger implementation that ignores all messages.
class EmptyLogger : public Logger {
public:
  static EmptyLogger &getInstance();

  void log(const llvm::Twine &Message) override;

private:
  EmptyLogger() = default;
};

} // namespace clangd
} // namespace clang

#endif
