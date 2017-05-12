//===-- NativeBreakpoint.h --------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_NativeBreakpoint_h_
#define liblldb_NativeBreakpoint_h_

#include "lldb/lldb-types.h"

namespace lldb_private {
class NativeBreakpointList;

class NativeBreakpoint {
  friend class NativeBreakpointList;

public:
  // The assumption is that derived breakpoints are enabled when created.
  NativeBreakpoint(lldb::addr_t addr);

  virtual ~NativeBreakpoint();

  Status Enable();

  Status Disable();

  lldb::addr_t GetAddress() const { return m_addr; }

  bool IsEnabled() const { return m_enabled; }

  virtual bool IsSoftwareBreakpoint() const = 0;

protected:
  const lldb::addr_t m_addr;
  int32_t m_ref_count;

  virtual Status DoEnable() = 0;

  virtual Status DoDisable() = 0;

private:
  bool m_enabled;

  // -----------------------------------------------------------
  // interface for NativeBreakpointList
  // -----------------------------------------------------------
  void AddRef();
  int32_t DecRef();
};
}

#endif // ifndef liblldb_NativeBreakpoint_h_
