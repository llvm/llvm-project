//===-- Diagnostics.h -------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_CORE_DIAGNOSTICS_H
#define LLDB_CORE_DIAGNOSTICS_H

#include "lldb/Utility/FileSpec.h"
#include "lldb/Utility/Log.h"
#include "llvm/Support/Error.h"

#include <optional>

namespace lldb_private {

/// Diagnostics maintain an always-on, in-memory log of recent diagnostic
/// messages that can be written out to help investigate bugs and troubleshoot
/// issues.
class Diagnostics {
public:
  Diagnostics();
  ~Diagnostics();

  /// Write the in-memory diagnostic log into the given directory.
  llvm::Error Create(const FileSpec &dir);

  /// Write the diagnostic log into a directory and print a message to the given
  /// output stream.
  /// @{
  bool Dump(llvm::raw_ostream &stream);
  bool Dump(llvm::raw_ostream &stream, const FileSpec &dir);
  /// @}

  void Report(llvm::StringRef message);

  static Diagnostics &Instance();

  static bool Enabled();
  static void Initialize();
  static void Terminate();

  /// Create a unique diagnostic directory.
  static llvm::Expected<FileSpec> CreateUniqueDirectory();

private:
  static std::optional<Diagnostics> &InstanceImpl();

  llvm::Error DumpDiangosticsLog(const FileSpec &dir) const;

  RotatingLogHandler m_log_handler;
};

} // namespace lldb_private

#endif
