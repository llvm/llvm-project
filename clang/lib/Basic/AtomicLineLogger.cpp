//===- AtomicLineLogger.cpp -----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the implementation of an AtomicLineLogger and the relevant
// supporting classes.
//
//===----------------------------------------------------------------------===//

#include "clang/Basic/AtomicLineLogger.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Errno.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/Process.h"
#include "llvm/Support/Threading.h"
#ifndef _WIN32
#include <unistd.h>
#endif
#ifdef __APPLE__
#include <sys/time.h>
#endif

using namespace clang;

static uint64_t getTimestampMillis() {
#ifdef __APPLE__
  // Using chrono is roughly 50% slower.
  struct timeval T;
  gettimeofday(&T, 0);
  return T.tv_sec * 1000 + T.tv_usec / 1000;
#else
  auto Time = std::chrono::system_clock::now();
  auto Millis = std::chrono::duration_cast<std::chrono::milliseconds>(
      Time.time_since_epoch());
  return Millis.count();
#endif
}

// Writes the whole line into an FD that is opened with OF_Append.
// This function only does one write (up to retry due to interrupts), and the
// single write is blocking and atomic on POSIX systems.
static bool writeLineToFD(int FD, const char *Data, size_t Size) {
#ifndef _WIN32
  ssize_t Written = llvm::sys::RetryAfterSignal(-1, write, FD, Data, Size);
  return Written >= 0 && (static_cast<size_t>(Written) == Size);
#else
  (void)FD, (void)Data, (void)Size;
  llvm_unreachable("Logging not supported on Windows!");
  return false;
#endif
}

LogLine::LogLine(int FD, std::atomic<uint64_t> *DroppedLines)
    : FormattingOS(Buffer), FD(FD), DroppedLines(DroppedLines) {
  auto Millis = getTimestampMillis();
  *FormattingOS << llvm::format("[%lld.%0.3lld]", Millis / 1000, Millis % 1000);
  *FormattingOS << ' ' << llvm::sys::Process::getProcessId() << ' '
                << llvm::get_threadid() << ": ";
}

LogLine::LogLine(LogLine &&Other)
    : Buffer(std::move(Other.Buffer)), FD(Other.FD),
      DroppedLines(Other.DroppedLines) {
  if (Other.FormattingOS)
    FormattingOS.emplace(Buffer);

  // Destroy the info in Other so its destructor does not write out the line.
  Other.FormattingOS.reset();
  Other.FD = -1;
  Other.DroppedLines = nullptr;
}

LogLine::~LogLine() {
  if (!FormattingOS)
    return;
  *FormattingOS << "\n";
  if (!writeLineToFD(FD, Buffer.data(), Buffer.size()))
    DroppedLines->fetch_add(1, std::memory_order_relaxed);
}

AtomicLineLogger::AtomicLineLogger(StringRef LogFilePath)
    : LogPath(LogFilePath.str()) {
#ifndef _WIN32
  if (LogFilePath.empty())
    return;

  std::error_code EC = llvm::sys::fs::openFileForWrite(
      LogFilePath, FD, llvm::sys::fs::CD_OpenAlways, llvm::sys::fs::OF_Append);
  if (EC) {
    llvm::errs() << "warning: unable to open log file '" << LogFilePath
                 << "': " << EC.message() << "\n";
    FD = -1;
    return;
  }
#endif
  // Write to files opened with OF_Append may not be guaranteed to be atomic
  // on Windows, so we do not enable logging on Windows.
}

LogLine AtomicLineLogger::log() {
  if (FD != -1)
    return LogLine(FD, &DroppedLines);
  return LogLine();
}

AtomicLineLogger::~AtomicLineLogger() {
  if (FD == -1)
    return;
  if (uint64_t Dropped = DroppedLines.load(std::memory_order_relaxed))
    llvm::errs() << "warning: log '" << LogPath
                 << "' is incomplete: " << Dropped
                 << " line(s) dropped due to write errors\n";
  llvm::sys::Process::SafelyCloseFileDescriptor(FD);
}
