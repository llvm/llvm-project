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

#include "lldb/Core/Error.h"
#include "lldb/Host/Mutex.h"
#include "lldb/lldb-private-forward.h"
// #include "lldb/Host/NativeBreakpoint.h"

#include <functional>
#include <map>

namespace lldb_private {
class NativeBreakpointList {
public:
  typedef std::function<Error(lldb::addr_t addr, size_t size_hint,
                              bool hardware, NativeBreakpointSP &breakpoint_sp)>
      CreateBreakpointFunc;

  NativeBreakpointList();

  Error AddRef(lldb::addr_t addr, size_t size_hint, bool hardware,
               CreateBreakpointFunc create_func);

  Error DecRef(lldb::addr_t addr);

  Error EnableBreakpoint(lldb::addr_t addr);

  Error DisableBreakpoint(lldb::addr_t addr);

  Error GetBreakpoint(lldb::addr_t addr, NativeBreakpointSP &breakpoint_sp);

  Error RemoveTrapsFromBuffer(lldb::addr_t addr, void *buf, size_t size) const;

private:
  typedef std::map<lldb::addr_t, NativeBreakpointSP> BreakpointMap;

  Mutex m_mutex;
  BreakpointMap m_breakpoints;
};
}

#endif // ifndef liblldb_NativeBreakpointList_h_
