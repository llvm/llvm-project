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
#include "llvm/Support/raw_ostream.h"
#include <memory>

namespace clang {

class LogLine {
  SmallString<128> Buffer;
  std::optional<llvm::raw_svector_ostream> FormattingOS;
  raw_ostream *Dest = nullptr;

public:
  explicit LogLine(raw_ostream &Dest);
  LogLine() {}
  LogLine(LogLine &&Other);
  LogLine(const LogLine &) = delete;
  LogLine &operator=(const LogLine &) = delete;
  LogLine &operator=(LogLine &&) = delete;

  ~LogLine() {
    if (Dest) {
      assert(FormattingOS && "Cannot have uninitialized FormattingOS");
      *FormattingOS << '\n';
      Dest->write(Buffer.data(), Buffer.size());
    }
  }

  template <typename T> LogLine &operator<<(const T &Val) {
    if (Dest) {
      assert(FormattingOS && "Cannot have uninitialized FormattingOS");
      *FormattingOS << Val;
    }
    return *this;
  }
};

class AtomicLineLogger {
  std::unique_ptr<llvm::raw_fd_ostream> OS;

public:
  AtomicLineLogger() {}
  AtomicLineLogger(StringRef LogFilePath);

  LogLine log();
};

} // namespace clang

#endif

