//===-- NativeThreadProtocol.h ----------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_NativeThreadProtocol_h_
#define liblldb_NativeThreadProtocol_h_

#include <memory>

#include "lldb/Host/Debug.h"
#include "lldb/lldb-private-forward.h"
#include "lldb/lldb-types.h"

namespace lldb_private {
//------------------------------------------------------------------
// NativeThreadProtocol
//------------------------------------------------------------------
class NativeThreadProtocol
    : public std::enable_shared_from_this<NativeThreadProtocol> {
public:
  NativeThreadProtocol(NativeProcessProtocol *process, lldb::tid_t tid);

  virtual ~NativeThreadProtocol() {}

  virtual std::string GetName() = 0;

  virtual lldb::StateType GetState() = 0;

  virtual NativeRegisterContextSP GetRegisterContext() = 0;

  virtual Status ReadRegister(uint32_t reg, RegisterValue &reg_value);

  virtual Status WriteRegister(uint32_t reg, const RegisterValue &reg_value);

  virtual Status SaveAllRegisters(lldb::DataBufferSP &data_sp);

  virtual Status RestoreAllRegisters(lldb::DataBufferSP &data_sp);

  virtual bool GetStopReason(ThreadStopInfo &stop_info,
                             std::string &description) = 0;

  lldb::tid_t GetID() const { return m_tid; }

  NativeProcessProtocolSP GetProcess();

  // ---------------------------------------------------------------------
  // Thread-specific watchpoints
  // ---------------------------------------------------------------------
  virtual Status SetWatchpoint(lldb::addr_t addr, size_t size,
                               uint32_t watch_flags, bool hardware) = 0;

  virtual Status RemoveWatchpoint(lldb::addr_t addr) = 0;

  // ---------------------------------------------------------------------
  // Thread-specific Hardware Breakpoint routines
  // ---------------------------------------------------------------------
  virtual Status SetHardwareBreakpoint(lldb::addr_t addr, size_t size) = 0;

  virtual Status RemoveHardwareBreakpoint(lldb::addr_t addr) = 0;

protected:
  NativeProcessProtocolWP m_process_wp;
  lldb::tid_t m_tid;
};
}

#endif // #ifndef liblldb_NativeThreadProtocol_h_
