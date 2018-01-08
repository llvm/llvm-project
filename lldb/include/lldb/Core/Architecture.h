//===-- Architecture.h ------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_CORE_ARCHITECTURE_H
#define LLDB_CORE_ARCHITECTURE_H

#include "lldb/Core/PluginInterface.h"

namespace lldb_private {

class Architecture : public PluginInterface {
public:
  Architecture() = default;
  virtual ~Architecture() = default;

  //------------------------------------------------------------------
  /// This is currently intended to handle cases where a
  /// program stops at an instruction that won't get executed and it
  /// allows the stop reason, like "breakpoint hit", to be replaced
  /// with a different stop reason like "no stop reason".
  ///
  /// This is specifically used for ARM in Thumb code when we stop in
  /// an IT instruction (if/then/else) where the instruction won't get
  /// executed and therefore it wouldn't be correct to show the program
  /// stopped at the current PC. The code is generic and applies to all
  /// ARM CPUs.
  //------------------------------------------------------------------
  virtual void OverrideStopInfo(Thread &thread) = 0;

private:
  Architecture(const Architecture &) = delete;
  void operator=(const Architecture &) = delete;
};

} // namespace lldb_private

#endif // LLDB_CORE_ARCHITECTURE_H
