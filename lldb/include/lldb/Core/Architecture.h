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

  //------------------------------------------------------------------
  /// This method is used to get the number of bytes that should be
  /// skipped, from function start address, to reach the first
  /// instruction after the prologue. If overrode, it must return
  /// non-zero only if the current address matches one of the known
  /// function entry points.
  ///
  /// This method is called only if the standard platform-independent
  /// code fails to get the number of bytes to skip, giving the plugin
  /// a chance to try to find the missing info.
  ///
  /// This is specifically used for PPC64, where functions may have
  /// more than one entry point, global and local, so both should
  /// be compared with current address, in order to find out the
  /// number of bytes that should be skipped, in case we are stopped
  /// at either function entry point.
  //------------------------------------------------------------------
  virtual size_t GetBytesToSkip(Symbol &func, const Address &curr_addr) const {
    return 0;
  }

  //------------------------------------------------------------------
  /// Adjust function breakpoint address, if needed. In some cases,
  /// the function start address is not the right place to set the
  /// breakpoint, specially in functions with multiple entry points.
  ///
  /// This is specifically used for PPC64, for functions that have
  /// both a global and a local entry point. In this case, the
  /// breakpoint is adjusted to the first function address reached
  /// by both entry points.
  //------------------------------------------------------------------
  virtual void AdjustBreakpointAddress(const Symbol &func,
                                       Address &addr) const {}

private:
  Architecture(const Architecture &) = delete;
  void operator=(const Architecture &) = delete;
};

} // namespace lldb_private

#endif // LLDB_CORE_ARCHITECTURE_H
