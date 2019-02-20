//===-- ProcessInfo.h -------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_ProcessInfo_h_
#define liblldb_ProcessInfo_h_

// LLDB headers
#include "lldb/Utility/ArchSpec.h"
#include "lldb/Utility/Args.h"
#include "lldb/Utility/Environment.h"
#include "lldb/Utility/FileSpec.h"

namespace lldb_private {
//----------------------------------------------------------------------
// ProcessInfo
//
// A base class for information for a process. This can be used to fill
// out information for a process prior to launching it, or it can be used for
// an instance of a process and can be filled in with the existing values for
// that process.
//----------------------------------------------------------------------
class ProcessInfo {
public:
  ProcessInfo();

  ProcessInfo(const char *name, const ArchSpec &arch, lldb::pid_t pid);

  void Clear();

  const char *GetName() const;

  size_t GetNameLength() const;

  FileSpec &GetExecutableFile() { return m_executable; }

  void SetExecutableFile(const FileSpec &exe_file,
                         bool add_exe_file_as_first_arg);

  const FileSpec &GetExecutableFile() const { return m_executable; }

  uint32_t GetUserID() const { return m_uid; }

  uint32_t GetGroupID() const { return m_gid; }

  bool UserIDIsValid() const { return m_uid != UINT32_MAX; }

  bool GroupIDIsValid() const { return m_gid != UINT32_MAX; }

  void SetUserID(uint32_t uid) { m_uid = uid; }

  void SetGroupID(uint32_t gid) { m_gid = gid; }

  ArchSpec &GetArchitecture() { return m_arch; }

  const ArchSpec &GetArchitecture() const { return m_arch; }

  void SetArchitecture(const ArchSpec &arch) { m_arch = arch; }

  lldb::pid_t GetProcessID() const { return m_pid; }

  void SetProcessID(lldb::pid_t pid) { m_pid = pid; }

  bool ProcessIDIsValid() const { return m_pid != LLDB_INVALID_PROCESS_ID; }

  void Dump(Stream &s, Platform *platform) const;

  Args &GetArguments() { return m_arguments; }

  const Args &GetArguments() const { return m_arguments; }

  llvm::StringRef GetArg0() const;

  void SetArg0(llvm::StringRef arg);

  void SetArguments(const Args &args, bool first_arg_is_executable);

  void SetArguments(char const **argv, bool first_arg_is_executable);

  Environment &GetEnvironment() { return m_environment; }
  const Environment &GetEnvironment() const { return m_environment; }

protected:
  FileSpec m_executable;
  std::string m_arg0; // argv[0] if supported. If empty, then use m_executable.
  // Not all process plug-ins support specifying an argv[0] that differs from
  // the resolved platform executable (which is in m_executable)
  Args m_arguments; // All program arguments except argv[0]
  Environment m_environment;
  uint32_t m_uid;
  uint32_t m_gid;
  ArchSpec m_arch;
  lldb::pid_t m_pid;
};
}

#endif // #ifndef liblldb_ProcessInfo_h_
