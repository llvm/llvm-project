//===-- HostProcessWindows.h ------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef lldb_Host_HostProcessWindows_h_
#define lldb_Host_HostProcessWindows_h_

#include "lldb/Host/HostNativeProcessBase.h"
#include "lldb/lldb-types.h"

namespace lldb_private {

class FileSpec;

class HostProcessWindows : public HostNativeProcessBase {
public:
  HostProcessWindows();
  explicit HostProcessWindows(lldb::process_t process);
  ~HostProcessWindows();

  void SetOwnsHandle(bool owns);

  Error Terminate() override;
  Error GetMainModule(FileSpec &file_spec) const override;

  lldb::pid_t GetProcessId() const override;
  bool IsRunning() const override;

  HostThread StartMonitoring(const Host::MonitorChildProcessCallback &callback,
                             bool monitor_signals) override;

private:
  static lldb::thread_result_t MonitorThread(void *thread_arg);

  void Close();

  bool m_owns_handle;
};
}

#endif
