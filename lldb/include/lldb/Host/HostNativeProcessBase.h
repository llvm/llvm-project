//===-- HostNativeProcessBase.h ---------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef lldb_Host_HostNativeProcessBase_h_
#define lldb_Host_HostNativeProcessBase_h_

#include "lldb/Host/HostProcess.h"
#include "lldb/Utility/Status.h"
#include "lldb/lldb-defines.h"
#include "lldb/lldb-types.h"

namespace lldb_private {

class HostThread;

class HostNativeProcessBase {
  DISALLOW_COPY_AND_ASSIGN(HostNativeProcessBase);

public:
  HostNativeProcessBase() : m_process(LLDB_INVALID_PROCESS) {}
  explicit HostNativeProcessBase(lldb::process_t process)
      : m_process(process) {}
  virtual ~HostNativeProcessBase() {}

  virtual Status Terminate() = 0;
  virtual Status GetMainModule(FileSpec &file_spec) const = 0;

  virtual lldb::pid_t GetProcessId() const = 0;
  virtual bool IsRunning() const = 0;

  lldb::process_t GetSystemHandle() const { return m_process; }

  virtual llvm::Expected<HostThread>
  StartMonitoring(const Host::MonitorChildProcessCallback &callback,
                  bool monitor_signals) = 0;

protected:
  lldb::process_t m_process;
};
}

#endif
