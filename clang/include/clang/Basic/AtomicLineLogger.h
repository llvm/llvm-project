//===- AtomicLineLogger.h ---------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file
/// Defines a logger where each line is written atomically to the file. It is
/// safe to share a logger instance across threads.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_BASIC_ATOMICLINELOGGER_H
#define LLVM_CLANG_BASIC_ATOMICLINELOGGER_H

#include "clang/Basic/LLVM.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Support/raw_ostream.h"
#include <atomic>
#include <optional>
#include <string>

namespace clang {

class LogLine {
  SmallString<128> Buffer;
  std::optional<llvm::raw_svector_ostream> FormattingOS;
  int FD = -1;
  std::atomic<uint64_t> *DroppedLines = nullptr;

  explicit LogLine(int FD, std::atomic<uint64_t> *DroppedLines);

public:
  LogLine() {}
  LogLine(LogLine &&Other);
  LogLine(const LogLine &) = delete;
  LogLine &operator=(const LogLine &) = delete;
  LogLine &operator=(LogLine &&) = delete;

  ~LogLine();

  template <typename T> LogLine &operator<<(const T &Val) {
    if (FormattingOS)
      *FormattingOS << Val;
    return *this;
  }

  template <typename RangeT>
  void logArray(StringRef Prefix, StringRef Sep, const RangeT &Arr) {
    if (FormattingOS) {
      *FormattingOS << Prefix;
      for (const auto &C : Arr)
        *FormattingOS << Sep << C;
    }
  }

  friend class AtomicLineLogger;
};

class AtomicLineLogger {
  int FD = -1;
  std::string LogPath;
  std::atomic<uint64_t> DroppedLines{0};

public:
  AtomicLineLogger() {}
  AtomicLineLogger(StringRef LogFilePath);

  // AtomicLineLogger is non-movable because LogLines have pointers to the
  // atomic member DroppedLines.
  AtomicLineLogger(AtomicLineLogger &&) = delete;
  AtomicLineLogger &operator=(AtomicLineLogger &&) = delete;

  ~AtomicLineLogger();

  LogLine log();
};

} // namespace clang

#endif
