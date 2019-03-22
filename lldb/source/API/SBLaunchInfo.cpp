//===-- SBLaunchInfo.cpp ----------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/API/SBLaunchInfo.h"
#include "SBReproducerPrivate.h"

#include "lldb/API/SBFileSpec.h"
#include "lldb/API/SBListener.h"
#include "lldb/Host/ProcessLaunchInfo.h"

using namespace lldb;
using namespace lldb_private;

class lldb_private::SBLaunchInfoImpl : public ProcessLaunchInfo {
public:
  SBLaunchInfoImpl()
      : ProcessLaunchInfo(), m_envp(GetEnvironment().getEnvp()) {}

  const char *const *GetEnvp() const { return m_envp; }
  void RegenerateEnvp() { m_envp = GetEnvironment().getEnvp(); }

  SBLaunchInfoImpl &operator=(const ProcessLaunchInfo &rhs) {
    ProcessLaunchInfo::operator=(rhs);
    RegenerateEnvp();
    return *this;
  }

private:
  Environment::Envp m_envp;
};

SBLaunchInfo::SBLaunchInfo(const char **argv)
    : m_opaque_sp(new SBLaunchInfoImpl()) {
  LLDB_RECORD_CONSTRUCTOR(SBLaunchInfo, (const char **), argv);

  m_opaque_sp->GetFlags().Reset(eLaunchFlagDebug | eLaunchFlagDisableASLR);
  if (argv && argv[0])
    m_opaque_sp->GetArguments().SetArguments(argv);
}

SBLaunchInfo::~SBLaunchInfo() {}

const lldb_private::ProcessLaunchInfo &SBLaunchInfo::ref() const {
  return *m_opaque_sp;
}

void SBLaunchInfo::set_ref(const ProcessLaunchInfo &info) {
  *m_opaque_sp = info;
}

lldb::pid_t SBLaunchInfo::GetProcessID() {
  LLDB_RECORD_METHOD_NO_ARGS(lldb::pid_t, SBLaunchInfo, GetProcessID);

  return m_opaque_sp->GetProcessID();
}

uint32_t SBLaunchInfo::GetUserID() {
  LLDB_RECORD_METHOD_NO_ARGS(uint32_t, SBLaunchInfo, GetUserID);

  return m_opaque_sp->GetUserID();
}

uint32_t SBLaunchInfo::GetGroupID() {
  LLDB_RECORD_METHOD_NO_ARGS(uint32_t, SBLaunchInfo, GetGroupID);

  return m_opaque_sp->GetGroupID();
}

bool SBLaunchInfo::UserIDIsValid() {
  LLDB_RECORD_METHOD_NO_ARGS(bool, SBLaunchInfo, UserIDIsValid);

  return m_opaque_sp->UserIDIsValid();
}

bool SBLaunchInfo::GroupIDIsValid() {
  LLDB_RECORD_METHOD_NO_ARGS(bool, SBLaunchInfo, GroupIDIsValid);

  return m_opaque_sp->GroupIDIsValid();
}

void SBLaunchInfo::SetUserID(uint32_t uid) {
  LLDB_RECORD_METHOD(void, SBLaunchInfo, SetUserID, (uint32_t), uid);

  m_opaque_sp->SetUserID(uid);
}

void SBLaunchInfo::SetGroupID(uint32_t gid) {
  LLDB_RECORD_METHOD(void, SBLaunchInfo, SetGroupID, (uint32_t), gid);

  m_opaque_sp->SetGroupID(gid);
}

SBFileSpec SBLaunchInfo::GetExecutableFile() {
  LLDB_RECORD_METHOD_NO_ARGS(lldb::SBFileSpec, SBLaunchInfo, GetExecutableFile);

  return LLDB_RECORD_RESULT(SBFileSpec(m_opaque_sp->GetExecutableFile()));
}

void SBLaunchInfo::SetExecutableFile(SBFileSpec exe_file,
                                     bool add_as_first_arg) {
  LLDB_RECORD_METHOD(void, SBLaunchInfo, SetExecutableFile,
                     (lldb::SBFileSpec, bool), exe_file, add_as_first_arg);

  m_opaque_sp->SetExecutableFile(exe_file.ref(), add_as_first_arg);
}

SBListener SBLaunchInfo::GetListener() {
  LLDB_RECORD_METHOD_NO_ARGS(lldb::SBListener, SBLaunchInfo, GetListener);

  return LLDB_RECORD_RESULT(SBListener(m_opaque_sp->GetListener()));
}

void SBLaunchInfo::SetListener(SBListener &listener) {
  LLDB_RECORD_METHOD(void, SBLaunchInfo, SetListener, (lldb::SBListener &),
                     listener);

  m_opaque_sp->SetListener(listener.GetSP());
}

uint32_t SBLaunchInfo::GetNumArguments() {
  LLDB_RECORD_METHOD_NO_ARGS(uint32_t, SBLaunchInfo, GetNumArguments);

  return m_opaque_sp->GetArguments().GetArgumentCount();
}

const char *SBLaunchInfo::GetArgumentAtIndex(uint32_t idx) {
  LLDB_RECORD_METHOD(const char *, SBLaunchInfo, GetArgumentAtIndex, (uint32_t),
                     idx);

  return m_opaque_sp->GetArguments().GetArgumentAtIndex(idx);
}

void SBLaunchInfo::SetArguments(const char **argv, bool append) {
  LLDB_RECORD_METHOD(void, SBLaunchInfo, SetArguments, (const char **, bool),
                     argv, append);

  if (append) {
    if (argv)
      m_opaque_sp->GetArguments().AppendArguments(argv);
  } else {
    if (argv)
      m_opaque_sp->GetArguments().SetArguments(argv);
    else
      m_opaque_sp->GetArguments().Clear();
  }
}

uint32_t SBLaunchInfo::GetNumEnvironmentEntries() {
  LLDB_RECORD_METHOD_NO_ARGS(uint32_t, SBLaunchInfo, GetNumEnvironmentEntries);

  return m_opaque_sp->GetEnvironment().size();
}

const char *SBLaunchInfo::GetEnvironmentEntryAtIndex(uint32_t idx) {
  LLDB_RECORD_METHOD(const char *, SBLaunchInfo, GetEnvironmentEntryAtIndex,
                     (uint32_t), idx);

  if (idx > GetNumEnvironmentEntries())
    return nullptr;
  return m_opaque_sp->GetEnvp()[idx];
}

void SBLaunchInfo::SetEnvironmentEntries(const char **envp, bool append) {
  LLDB_RECORD_METHOD(void, SBLaunchInfo, SetEnvironmentEntries,
                     (const char **, bool), envp, append);

  Environment env(envp);
  if (append)
    m_opaque_sp->GetEnvironment().insert(env.begin(), env.end());
  else
    m_opaque_sp->GetEnvironment() = env;
  m_opaque_sp->RegenerateEnvp();
}

void SBLaunchInfo::Clear() {
  LLDB_RECORD_METHOD_NO_ARGS(void, SBLaunchInfo, Clear);

  m_opaque_sp->Clear();
}

const char *SBLaunchInfo::GetWorkingDirectory() const {
  LLDB_RECORD_METHOD_CONST_NO_ARGS(const char *, SBLaunchInfo,
                                   GetWorkingDirectory);

  return m_opaque_sp->GetWorkingDirectory().GetCString();
}

void SBLaunchInfo::SetWorkingDirectory(const char *working_dir) {
  LLDB_RECORD_METHOD(void, SBLaunchInfo, SetWorkingDirectory, (const char *),
                     working_dir);

  m_opaque_sp->SetWorkingDirectory(FileSpec(working_dir));
}

uint32_t SBLaunchInfo::GetLaunchFlags() {
  LLDB_RECORD_METHOD_NO_ARGS(uint32_t, SBLaunchInfo, GetLaunchFlags);

  return m_opaque_sp->GetFlags().Get();
}

void SBLaunchInfo::SetLaunchFlags(uint32_t flags) {
  LLDB_RECORD_METHOD(void, SBLaunchInfo, SetLaunchFlags, (uint32_t), flags);

  m_opaque_sp->GetFlags().Reset(flags);
}

const char *SBLaunchInfo::GetProcessPluginName() {
  LLDB_RECORD_METHOD_NO_ARGS(const char *, SBLaunchInfo, GetProcessPluginName);

  return m_opaque_sp->GetProcessPluginName();
}

void SBLaunchInfo::SetProcessPluginName(const char *plugin_name) {
  LLDB_RECORD_METHOD(void, SBLaunchInfo, SetProcessPluginName, (const char *),
                     plugin_name);

  return m_opaque_sp->SetProcessPluginName(plugin_name);
}

const char *SBLaunchInfo::GetShell() {
  LLDB_RECORD_METHOD_NO_ARGS(const char *, SBLaunchInfo, GetShell);

  // Constify this string so that it is saved in the string pool.  Otherwise it
  // would be freed when this function goes out of scope.
  ConstString shell(m_opaque_sp->GetShell().GetPath().c_str());
  return shell.AsCString();
}

void SBLaunchInfo::SetShell(const char *path) {
  LLDB_RECORD_METHOD(void, SBLaunchInfo, SetShell, (const char *), path);

  m_opaque_sp->SetShell(FileSpec(path));
}

bool SBLaunchInfo::GetShellExpandArguments() {
  LLDB_RECORD_METHOD_NO_ARGS(bool, SBLaunchInfo, GetShellExpandArguments);

  return m_opaque_sp->GetShellExpandArguments();
}

void SBLaunchInfo::SetShellExpandArguments(bool expand) {
  LLDB_RECORD_METHOD(void, SBLaunchInfo, SetShellExpandArguments, (bool),
                     expand);

  m_opaque_sp->SetShellExpandArguments(expand);
}

uint32_t SBLaunchInfo::GetResumeCount() {
  LLDB_RECORD_METHOD_NO_ARGS(uint32_t, SBLaunchInfo, GetResumeCount);

  return m_opaque_sp->GetResumeCount();
}

void SBLaunchInfo::SetResumeCount(uint32_t c) {
  LLDB_RECORD_METHOD(void, SBLaunchInfo, SetResumeCount, (uint32_t), c);

  m_opaque_sp->SetResumeCount(c);
}

bool SBLaunchInfo::AddCloseFileAction(int fd) {
  LLDB_RECORD_METHOD(bool, SBLaunchInfo, AddCloseFileAction, (int), fd);

  return m_opaque_sp->AppendCloseFileAction(fd);
}

bool SBLaunchInfo::AddDuplicateFileAction(int fd, int dup_fd) {
  LLDB_RECORD_METHOD(bool, SBLaunchInfo, AddDuplicateFileAction, (int, int), fd,
                     dup_fd);

  return m_opaque_sp->AppendDuplicateFileAction(fd, dup_fd);
}

bool SBLaunchInfo::AddOpenFileAction(int fd, const char *path, bool read,
                                     bool write) {
  LLDB_RECORD_METHOD(bool, SBLaunchInfo, AddOpenFileAction,
                     (int, const char *, bool, bool), fd, path, read, write);

  return m_opaque_sp->AppendOpenFileAction(fd, FileSpec(path), read, write);
}

bool SBLaunchInfo::AddSuppressFileAction(int fd, bool read, bool write) {
  LLDB_RECORD_METHOD(bool, SBLaunchInfo, AddSuppressFileAction,
                     (int, bool, bool), fd, read, write);

  return m_opaque_sp->AppendSuppressFileAction(fd, read, write);
}

void SBLaunchInfo::SetLaunchEventData(const char *data) {
  LLDB_RECORD_METHOD(void, SBLaunchInfo, SetLaunchEventData, (const char *),
                     data);

  m_opaque_sp->SetLaunchEventData(data);
}

const char *SBLaunchInfo::GetLaunchEventData() const {
  LLDB_RECORD_METHOD_CONST_NO_ARGS(const char *, SBLaunchInfo,
                                   GetLaunchEventData);

  return m_opaque_sp->GetLaunchEventData();
}

void SBLaunchInfo::SetDetachOnError(bool enable) {
  LLDB_RECORD_METHOD(void, SBLaunchInfo, SetDetachOnError, (bool), enable);

  m_opaque_sp->SetDetachOnError(enable);
}

bool SBLaunchInfo::GetDetachOnError() const {
  LLDB_RECORD_METHOD_CONST_NO_ARGS(bool, SBLaunchInfo, GetDetachOnError);

  return m_opaque_sp->GetDetachOnError();
}
