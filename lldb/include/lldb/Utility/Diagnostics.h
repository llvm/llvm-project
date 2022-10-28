//===-- Diagnostics.h -------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_UTILITY_DIAGNOSTICS_H
#define LLDB_UTILITY_DIAGNOSTICS_H

#include "lldb/Utility/FileSpec.h"
#include "lldb/Utility/Log.h"
#include "llvm/ADT/Optional.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/Error.h"

#include <functional>
#include <mutex>
#include <vector>

namespace lldb_private {

/// Diagnostics are a collection of files to help investigate bugs and
/// troubleshoot issues. Any part of the debugger can register itself with the
/// help of a callback to emit one or more files into the diagnostic directory.
class Diagnostics {
public:
  Diagnostics();
  ~Diagnostics();

  /// Gather diagnostics in the given directory.
  llvm::Error Create(const FileSpec &dir);

  /// Gather diagnostics and print a message to the given output stream.
  bool Dump(llvm::raw_ostream &stream);

  using Callback = std::function<llvm::Error(const FileSpec &)>;

  void AddCallback(Callback callback);

  static Diagnostics &Instance();
  static void Initialize();
  static void Terminate();

private:
  static llvm::Optional<Diagnostics> &InstanceImpl();

  llvm::SmallVector<Callback, 4> m_callbacks;
  std::mutex m_callbacks_mutex;
};

} // namespace lldb_private

#endif
