//===-- NativeBreakpointList.h ----------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_NativeBreakpointList_h_
#define liblldb_NativeBreakpointList_h_

#include "lldb/Utility/Status.h"
#include "lldb/lldb-private-forward.h"
// #include "lldb/Host/NativeBreakpoint.h"

#include <functional>
#include <map>
#include <mutex>

namespace lldb_private {

struct HardwareBreakpoint {
  lldb::addr_t m_addr;
  size_t m_size;
};

using HardwareBreakpointMap = std::map<lldb::addr_t, HardwareBreakpoint>;

class NativeBreakpointList {
public:
  typedef std::function<Status(lldb::addr_t addr, size_t size_hint,
                               bool hardware,
                               NativeBreakpointSP &breakpoint_sp)>
      CreateBreakpointFunc;

  NativeBreakpointList();

  Status AddRef(lldb::addr_t addr, size_t size_hint, bool hardware,
                CreateBreakpointFunc create_func);

  Status DecRef(lldb::addr_t addr);

  Status EnableBreakpoint(lldb::addr_t addr);

  Status DisableBreakpoint(lldb::addr_t addr);

  Status GetBreakpoint(lldb::addr_t addr, NativeBreakpointSP &breakpoint_sp);

  Status RemoveTrapsFromBuffer(lldb::addr_t addr, void *buf, size_t size) const;

private:
  typedef std::map<lldb::addr_t, NativeBreakpointSP> BreakpointMap;

  std::recursive_mutex m_mutex;
  BreakpointMap m_breakpoints;
};
}

#endif // ifndef liblldb_NativeBreakpointList_h_
