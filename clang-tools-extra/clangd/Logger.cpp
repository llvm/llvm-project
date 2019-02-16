//===--- Logger.cpp - Logger interface for clangd -------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "Logger.h"
#include "Trace.h"
#include "llvm/Support/Chrono.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/raw_ostream.h"
#include <mutex>

using namespace llvm;
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

void detail::log(Logger::Level Level, const formatv_object_base &Message) {
  if (L)
    L->log(Level, Message);
  else {
    static std::mutex Mu;
    std::lock_guard<std::mutex> Guard(Mu);
    errs() << Message << "\n";
  }
}

const char *detail::debugType(const char *Filename) {
  if (const char *Slash = strrchr(Filename, '/'))
    return Slash + 1;
  if (const char *Backslash = strrchr(Filename, '\\'))
    return Backslash + 1;
  return Filename;
}

void StreamLogger::log(Logger::Level Level,
                       const formatv_object_base &Message) {
  if (Level < MinLevel)
    return;
  sys::TimePoint<> Timestamp = std::chrono::system_clock::now();
  trace::log(Message);
  std::lock_guard<std::mutex> Guard(StreamMutex);
  Logs << formatv("{0}[{1:%H:%M:%S.%L}] {2}\n", indicator(Level), Timestamp,
                  Message);
  Logs.flush();
}

} // namespace clangd
} // namespace clang
