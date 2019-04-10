//===-- NativeThreadProtocol.h ----------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_NativeThreadProtocol_h_
#define liblldb_NativeThreadProtocol_h_

#include <memory>

#include "lldb/Host/Debug.h"
#include "lldb/lldb-private-forward.h"
#include "lldb/lldb-types.h"

namespace lldb_private {
// NativeThreadProtocol
class NativeThreadProtocol {
public:
  NativeThreadProtocol(NativeProcessProtocol &process, lldb::tid_t tid);

  virtual ~NativeThreadProtocol() {}

  virtual std::string GetName() = 0;

  virtual lldb::StateType GetState() = 0;

  virtual NativeRegisterContext &GetRegisterContext() = 0;

  virtual bool GetStopReason(ThreadStopInfo &stop_info,
                             std::string &description) = 0;

  lldb::tid_t GetID() const { return m_tid; }

  NativeProcessProtocol &GetProcess() { return m_process; }

  // Thread-specific watchpoints
  virtual Status SetWatchpoint(lldb::addr_t addr, size_t size,
                               uint32_t watch_flags, bool hardware) = 0;

  virtual Status RemoveWatchpoint(lldb::addr_t addr) = 0;

  // Thread-specific Hardware Breakpoint routines
  virtual Status SetHardwareBreakpoint(lldb::addr_t addr, size_t size) = 0;

  virtual Status RemoveHardwareBreakpoint(lldb::addr_t addr) = 0;

protected:
  NativeProcessProtocol &m_process;
  lldb::tid_t m_tid;
};
}

#endif // #ifndef liblldb_NativeThreadProtocol_h_
