//===-- RemoteAwarePlatform.h -----------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_TARGET_REMOTEAWAREPLATFORM_H
#define LLDB_TARGET_REMOTEAWAREPLATFORM_H

#include "lldb/Target/Platform.h"

namespace lldb_private {

/// A base class for platforms which automatically want to be able to forward
/// operations to a remote platform instance (such as PlatformRemoteGDBServer).
class RemoteAwarePlatform : public Platform {
public:
  using Platform::Platform;

  bool GetModuleSpec(const FileSpec &module_file_spec, const ArchSpec &arch,
                     ModuleSpec &module_spec) override;
  Status GetFileWithUUID(const FileSpec &platform_file, const UUID *uuid,
                         FileSpec &local_file) override;

  bool GetRemoteOSVersion() override;
  bool GetRemoteOSBuildString(std::string &s) override;
  bool GetRemoteOSKernelDescription(std::string &s) override;
  ArchSpec GetRemoteSystemArchitecture() override;

  const char *GetHostname() override;
  const char *GetUserName(uint32_t uid) override;
  const char *GetGroupName(uint32_t gid) override;
  lldb_private::Environment GetEnvironment() override;

  bool IsConnected() const override;

  bool GetProcessInfo(lldb::pid_t pid, ProcessInstanceInfo &proc_info) override;
  uint32_t FindProcesses(const ProcessInstanceInfoMatch &match_info,
                         ProcessInstanceInfoList &process_infos) override;
  Status LaunchProcess(ProcessLaunchInfo &launch_info) override;

protected:
  lldb::PlatformSP m_remote_platform_sp;
};

} // namespace lldb_private

#endif // LLDB_TARGET_REMOTEAWAREPLATFORM_H
