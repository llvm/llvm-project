//===-- PostMortemProcess.h -------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_TARGET_POSTMORTEMPROCESS_H
#define LLDB_TARGET_POSTMORTEMPROCESS_H

#include "lldb/Target/Process.h"

namespace lldb_private {

/// \class PostMortemProcess
/// Base class for all processes that don't represent a live process, such as
/// coredumps or processes traced in the past.
///
/// \a lldb_private::Process virtual functions overrides that are common
/// between these kinds of processes can have default implementations in this
/// class.
class PostMortemProcess : public Process {
public:
  using Process::Process;

  bool IsLiveDebugSession() const override { return false; }
};

} // namespace lldb_private

#endif // LLDB_TARGET_POSTMORTEMPROCESS_H
