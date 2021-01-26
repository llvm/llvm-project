//===-- ProcessTrace.h ------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_TARGET_PROCESSTRACE_H
#define LLDB_TARGET_PROCESSTRACE_H

#include "lldb/Target/PostMortemProcess.h"
#include "lldb/Utility/ConstString.h"
#include "lldb/Utility/Status.h"

namespace lldb_private {

class ProcessTrace : public PostMortemProcess {
public:
  static void Initialize();

  static void Terminate();

  static ConstString GetPluginNameStatic();

  static const char *GetPluginDescriptionStatic();

  ProcessTrace(lldb::TargetSP target_sp, lldb::ListenerSP listener_sp);

  ~ProcessTrace() override;

  bool CanDebug(lldb::TargetSP target_sp,
                bool plugin_specified_by_name) override;

  void DidAttach(ArchSpec &process_arch) override;

  DynamicLoader *GetDynamicLoader() override { return nullptr; }

  SystemRuntime *GetSystemRuntime() override { return nullptr; }

  ConstString GetPluginName() override;

  uint32_t GetPluginVersion() override;

  Status DoDestroy() override;

  void RefreshStateAfterStop() override;

  Status WillResume() override {
    Status error;
    error.SetErrorStringWithFormat(
        "error: %s does not support resuming processes",
        GetPluginName().GetCString());
    return error;
  }

  bool IsAlive() override;

  bool WarnBeforeDetach() const override { return false; }

  size_t ReadMemory(lldb::addr_t addr, void *buf, size_t size,
                    Status &error) override;

  size_t DoReadMemory(lldb::addr_t addr, void *buf, size_t size,
                      Status &error) override;

  ArchSpec GetArchitecture();

  bool GetProcessInfo(ProcessInstanceInfo &info) override;

protected:
  void Clear();

  bool DoUpdateThreadList(ThreadList &old_thread_list,
                          ThreadList &new_thread_list) override;

private:
  static lldb::ProcessSP CreateInstance(lldb::TargetSP target_sp,
                                        lldb::ListenerSP listener_sp,
                                        const FileSpec *crash_file_path,
                                        bool can_connect);
};

} // namespace lldb_private

#endif // LLDB_TARGET_PROCESSTRACE_H
