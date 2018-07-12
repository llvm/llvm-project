//===--- Logger.cpp - Logger interface for clangd -------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "Logger.h"
#include "llvm/Support/raw_ostream.h"
#include <mutex>

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

void detail::log(Logger::Level Level,
                 const llvm::formatv_object_base &Message) {
  if (L)
    L->log(Level, Message);
  else {
    static std::mutex Mu;
    std::lock_guard<std::mutex> Guard(Mu);
    llvm::errs() << Message << "\n";
  }
}

const char *detail::debugType(const char *Filename) {
  if (const char *Slash = strrchr(Filename, '/'))
    return Slash + 1;
  if (const char *Backslash = strrchr(Filename, '\\'))
    return Backslash + 1;
  return Filename;
}

} // namespace clangd
} // namespace clang
