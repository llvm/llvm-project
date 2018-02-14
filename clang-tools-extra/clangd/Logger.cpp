//===--- Logger.cpp - Logger interface for clangd -------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "Logger.h"

namespace clang {
namespace clangd {

namespace {
Logger *L = nullptr;
} // namespace

LoggingSession::LoggingSession(clangd::Logger &Instance) {
  assert(!L);
  L = &Instance;
}

LoggingSession::~LoggingSession() { L = nullptr; }

void log(const Context &Ctx, const llvm::Twine &Message) {
  if (!L)
    return;
  L->log(Ctx, Message);
}

} // namespace clangd
} // namespace clang
