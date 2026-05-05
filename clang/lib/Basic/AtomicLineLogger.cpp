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
#include "llvm/Support/Process.h"
#include "llvm/Support/Threading.h"
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

LogLine::LogLine(raw_ostream &Dest) : FormattingOS(Buffer), Dest(&Dest) {
  auto Millis = getTimestampMillis();
  assert(FormattingOS && "Cannot have unintialized FormattingOS");
  *FormattingOS << llvm::format("[%lld.%0.3lld]", Millis / 1000, Millis % 1000);
  *FormattingOS << ' ' << llvm::sys::Process::getProcessId() << ' '
                << llvm::get_threadid() << ": ";
}

LogLine::LogLine(LogLine &&Other)
    : Buffer(std::move(Other.Buffer)), Dest(Other.Dest) {
  if (Dest)
    FormattingOS.emplace(Buffer);
  Other.Dest = nullptr;
}

AtomicLineLogger::AtomicLineLogger(StringRef LogFilePath) {
  if (LogFilePath.empty())
    return;

  std::error_code EC;
  OS = std::make_unique<llvm::raw_fd_ostream>(
      LogFilePath, EC, llvm::sys::fs::CD_OpenAlways, llvm::sys::fs::FA_Write,
      llvm::sys::fs::OF_Append);
  if (EC) {
    llvm::errs() << "warning: unable to open log file '" << LogFilePath
                 << "': " << EC.message() << "\n";
    OS.reset();
    return;
  }

  // We need to set the OS to unbuffered, so LogLine's destructor can write
  // a single line as an atomic operation.
  OS->SetUnbuffered();
}

LogLine AtomicLineLogger::log() {
  if (OS)
    return LogLine(*OS);
  return LogLine();
}

