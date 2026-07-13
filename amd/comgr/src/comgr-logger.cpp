//===- comgr-logger.cpp - Global Comgr logging facility -------------------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file implements COMGR::Logger. See comgr-logger.h for the design.
///
//===----------------------------------------------------------------------===//

#include "comgr-logger.h"
#include "comgr-env.h"

#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/FileSystem.h"

using namespace llvm;

namespace COMGR {

namespace {

// Per-thread so a captured Action on one thread does not collect log output
// emitted by an unrelated API on another thread.
thread_local raw_ostream *ThreadCaptureStream = nullptr;

} // namespace

Logger::Logger() : Level(env::resolveLogLevel()), Sink(nullptr) {
  std::optional<StringRef> RedirectLogs = env::getRedirectLogs();
  if (RedirectLogs)
    openSink(*RedirectLogs);
}

Logger::Logger(LogLevel Level, raw_ostream *Sink) : Level(Level), Sink(Sink) {}

Logger::Logger(LogLevel Level, StringRef RedirectTarget)
    : Level(Level), Sink(nullptr) {
  openSink(RedirectTarget);
}

void Logger::openSink(StringRef RedirectLog) {
  if (RedirectLog == "stdout" || RedirectLog == "-") {
    Sink = &outs();
  } else if (RedirectLog == "stderr") {
    Sink = &errs();
  } else {
    std::error_code EC;
    SinkFile = std::make_unique<raw_fd_ostream>(
        RedirectLog, EC, sys::fs::OF_Text | sys::fs::OF_Append);
    if (EC) {
      SinkFile.reset();
      // Record rather than print: the action layer surfaces this into the
      // returned comgr.log via getSinkError().
      SinkError = (Twine("unable to redirect log to file '") + RedirectLog +
                   "': " + EC.message())
                      .str();
    } else {
      Sink = SinkFile.get();
      SinkFilename = RedirectLog.str();
    }
  }
}

void Logger::writeToSink(StringRef Data) {
  if (!Sink)
    return;

  std::scoped_lock<std::mutex> Lock(Mutex);
  *Sink << Data;
}

void Logger::sinkFlush() {
  if (!Sink)
    return;
  std::scoped_lock<std::mutex> Lock(Mutex);
  Sink->flush();
}

void Logger::emit(LogLevel Severity, const Twine &Message) {
  if (!isEnabled(Severity))
    return;

  SmallString<256> Buffer;
  StringRef Text = Message.toStringRef(Buffer);
  StringRef Prefix = "comgr: ";

  std::scoped_lock<std::mutex> Lock(Mutex);

  raw_ostream *Capture = ThreadCaptureStream;
  if (Sink) {
    *Sink << Prefix << Text << '\n';
    Sink->flush();
  }
  // Avoid double-emission if the capture stream aliases the sink.
  if (Capture && Capture != Sink) {
    *Capture << Prefix << Text << '\n';
    Capture->flush();
  }
}

Logger &getLogger() {
  static Logger TheLogger;
  return TheLogger;
}

raw_ostream *getThreadCaptureStream() { return ThreadCaptureStream; }

LogCaptureScope::LogCaptureScope(raw_ostream &OS)
    : Previous(ThreadCaptureStream) {
  ThreadCaptureStream = &OS;
}

LogCaptureScope::~LogCaptureScope() { ThreadCaptureStream = Previous; }

} // namespace COMGR
