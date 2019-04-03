//===-- SBAttachInfo.cpp ----------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/API/SBAttachInfo.h"
#include "SBReproducerPrivate.h"
#include "Utils.h"
#include "lldb/API/SBFileSpec.h"
#include "lldb/API/SBListener.h"
#include "lldb/Target/Process.h"

using namespace lldb;
using namespace lldb_private;

SBAttachInfo::SBAttachInfo() : m_opaque_sp(new ProcessAttachInfo()) {
  LLDB_RECORD_CONSTRUCTOR_NO_ARGS(SBAttachInfo);
}

SBAttachInfo::SBAttachInfo(lldb::pid_t pid)
    : m_opaque_sp(new ProcessAttachInfo()) {
  LLDB_RECORD_CONSTRUCTOR(SBAttachInfo, (lldb::pid_t), pid);

  m_opaque_sp->SetProcessID(pid);
}

SBAttachInfo::SBAttachInfo(const char *path, bool wait_for)
    : m_opaque_sp(new ProcessAttachInfo()) {
  LLDB_RECORD_CONSTRUCTOR(SBAttachInfo, (const char *, bool), path, wait_for);

  if (path && path[0])
    m_opaque_sp->GetExecutableFile().SetFile(path, FileSpec::Style::native);
  m_opaque_sp->SetWaitForLaunch(wait_for);
}

SBAttachInfo::SBAttachInfo(const char *path, bool wait_for, bool async)
    : m_opaque_sp(new ProcessAttachInfo()) {
  LLDB_RECORD_CONSTRUCTOR(SBAttachInfo, (const char *, bool, bool), path,
                          wait_for, async);

  if (path && path[0])
    m_opaque_sp->GetExecutableFile().SetFile(path, FileSpec::Style::native);
  m_opaque_sp->SetWaitForLaunch(wait_for);
  m_opaque_sp->SetAsync(async);
}

SBAttachInfo::SBAttachInfo(const SBAttachInfo &rhs)
    : m_opaque_sp(new ProcessAttachInfo()) {
  LLDB_RECORD_CONSTRUCTOR(SBAttachInfo, (const lldb::SBAttachInfo &), rhs);

  m_opaque_sp = clone(rhs.m_opaque_sp);
}

SBAttachInfo::~SBAttachInfo() {}

lldb_private::ProcessAttachInfo &SBAttachInfo::ref() { return *m_opaque_sp; }

SBAttachInfo &SBAttachInfo::operator=(const SBAttachInfo &rhs) {
  LLDB_RECORD_METHOD(lldb::SBAttachInfo &,
                     SBAttachInfo, operator=,(const lldb::SBAttachInfo &), rhs);

  if (this != &rhs)
    m_opaque_sp = clone(rhs.m_opaque_sp);
  return LLDB_RECORD_RESULT(*this);
}

lldb::pid_t SBAttachInfo::GetProcessID() {
  LLDB_RECORD_METHOD_NO_ARGS(lldb::pid_t, SBAttachInfo, GetProcessID);

  return m_opaque_sp->GetProcessID();
}

void SBAttachInfo::SetProcessID(lldb::pid_t pid) {
  LLDB_RECORD_METHOD(void, SBAttachInfo, SetProcessID, (lldb::pid_t), pid);

  m_opaque_sp->SetProcessID(pid);
}

uint32_t SBAttachInfo::GetResumeCount() {
  LLDB_RECORD_METHOD_NO_ARGS(uint32_t, SBAttachInfo, GetResumeCount);

  return m_opaque_sp->GetResumeCount();
}

void SBAttachInfo::SetResumeCount(uint32_t c) {
  LLDB_RECORD_METHOD(void, SBAttachInfo, SetResumeCount, (uint32_t), c);

  m_opaque_sp->SetResumeCount(c);
}

const char *SBAttachInfo::GetProcessPluginName() {
  LLDB_RECORD_METHOD_NO_ARGS(const char *, SBAttachInfo, GetProcessPluginName);

  return m_opaque_sp->GetProcessPluginName();
}

void SBAttachInfo::SetProcessPluginName(const char *plugin_name) {
  LLDB_RECORD_METHOD(void, SBAttachInfo, SetProcessPluginName, (const char *),
                     plugin_name);

  return m_opaque_sp->SetProcessPluginName(plugin_name);
}

void SBAttachInfo::SetExecutable(const char *path) {
  LLDB_RECORD_METHOD(void, SBAttachInfo, SetExecutable, (const char *), path);

  if (path && path[0])
    m_opaque_sp->GetExecutableFile().SetFile(path, FileSpec::Style::native);
  else
    m_opaque_sp->GetExecutableFile().Clear();
}

void SBAttachInfo::SetExecutable(SBFileSpec exe_file) {
  LLDB_RECORD_METHOD(void, SBAttachInfo, SetExecutable, (lldb::SBFileSpec),
                     exe_file);

  if (exe_file.IsValid())
    m_opaque_sp->GetExecutableFile() = exe_file.ref();
  else
    m_opaque_sp->GetExecutableFile().Clear();
}

bool SBAttachInfo::GetWaitForLaunch() {
  LLDB_RECORD_METHOD_NO_ARGS(bool, SBAttachInfo, GetWaitForLaunch);

  return m_opaque_sp->GetWaitForLaunch();
}

void SBAttachInfo::SetWaitForLaunch(bool b) {
  LLDB_RECORD_METHOD(void, SBAttachInfo, SetWaitForLaunch, (bool), b);

  m_opaque_sp->SetWaitForLaunch(b);
}

void SBAttachInfo::SetWaitForLaunch(bool b, bool async) {
  LLDB_RECORD_METHOD(void, SBAttachInfo, SetWaitForLaunch, (bool, bool), b,
                     async);

  m_opaque_sp->SetWaitForLaunch(b);
  m_opaque_sp->SetAsync(async);
}

bool SBAttachInfo::GetIgnoreExisting() {
  LLDB_RECORD_METHOD_NO_ARGS(bool, SBAttachInfo, GetIgnoreExisting);

  return m_opaque_sp->GetIgnoreExisting();
}

void SBAttachInfo::SetIgnoreExisting(bool b) {
  LLDB_RECORD_METHOD(void, SBAttachInfo, SetIgnoreExisting, (bool), b);

  m_opaque_sp->SetIgnoreExisting(b);
}

uint32_t SBAttachInfo::GetUserID() {
  LLDB_RECORD_METHOD_NO_ARGS(uint32_t, SBAttachInfo, GetUserID);

  return m_opaque_sp->GetUserID();
}

uint32_t SBAttachInfo::GetGroupID() {
  LLDB_RECORD_METHOD_NO_ARGS(uint32_t, SBAttachInfo, GetGroupID);

  return m_opaque_sp->GetGroupID();
}

bool SBAttachInfo::UserIDIsValid() {
  LLDB_RECORD_METHOD_NO_ARGS(bool, SBAttachInfo, UserIDIsValid);

  return m_opaque_sp->UserIDIsValid();
}

bool SBAttachInfo::GroupIDIsValid() {
  LLDB_RECORD_METHOD_NO_ARGS(bool, SBAttachInfo, GroupIDIsValid);

  return m_opaque_sp->GroupIDIsValid();
}

void SBAttachInfo::SetUserID(uint32_t uid) {
  LLDB_RECORD_METHOD(void, SBAttachInfo, SetUserID, (uint32_t), uid);

  m_opaque_sp->SetUserID(uid);
}

void SBAttachInfo::SetGroupID(uint32_t gid) {
  LLDB_RECORD_METHOD(void, SBAttachInfo, SetGroupID, (uint32_t), gid);

  m_opaque_sp->SetGroupID(gid);
}

uint32_t SBAttachInfo::GetEffectiveUserID() {
  LLDB_RECORD_METHOD_NO_ARGS(uint32_t, SBAttachInfo, GetEffectiveUserID);

  return m_opaque_sp->GetEffectiveUserID();
}

uint32_t SBAttachInfo::GetEffectiveGroupID() {
  LLDB_RECORD_METHOD_NO_ARGS(uint32_t, SBAttachInfo, GetEffectiveGroupID);

  return m_opaque_sp->GetEffectiveGroupID();
}

bool SBAttachInfo::EffectiveUserIDIsValid() {
  LLDB_RECORD_METHOD_NO_ARGS(bool, SBAttachInfo, EffectiveUserIDIsValid);

  return m_opaque_sp->EffectiveUserIDIsValid();
}

bool SBAttachInfo::EffectiveGroupIDIsValid() {
  LLDB_RECORD_METHOD_NO_ARGS(bool, SBAttachInfo, EffectiveGroupIDIsValid);

  return m_opaque_sp->EffectiveGroupIDIsValid();
}

void SBAttachInfo::SetEffectiveUserID(uint32_t uid) {
  LLDB_RECORD_METHOD(void, SBAttachInfo, SetEffectiveUserID, (uint32_t), uid);

  m_opaque_sp->SetEffectiveUserID(uid);
}

void SBAttachInfo::SetEffectiveGroupID(uint32_t gid) {
  LLDB_RECORD_METHOD(void, SBAttachInfo, SetEffectiveGroupID, (uint32_t), gid);

  m_opaque_sp->SetEffectiveGroupID(gid);
}

lldb::pid_t SBAttachInfo::GetParentProcessID() {
  LLDB_RECORD_METHOD_NO_ARGS(lldb::pid_t, SBAttachInfo, GetParentProcessID);

  return m_opaque_sp->GetParentProcessID();
}

void SBAttachInfo::SetParentProcessID(lldb::pid_t pid) {
  LLDB_RECORD_METHOD(void, SBAttachInfo, SetParentProcessID, (lldb::pid_t),
                     pid);

  m_opaque_sp->SetParentProcessID(pid);
}

bool SBAttachInfo::ParentProcessIDIsValid() {
  LLDB_RECORD_METHOD_NO_ARGS(bool, SBAttachInfo, ParentProcessIDIsValid);

  return m_opaque_sp->ParentProcessIDIsValid();
}

SBListener SBAttachInfo::GetListener() {
  LLDB_RECORD_METHOD_NO_ARGS(lldb::SBListener, SBAttachInfo, GetListener);

  return LLDB_RECORD_RESULT(SBListener(m_opaque_sp->GetListener()));
}

void SBAttachInfo::SetListener(SBListener &listener) {
  LLDB_RECORD_METHOD(void, SBAttachInfo, SetListener, (lldb::SBListener &),
                     listener);

  m_opaque_sp->SetListener(listener.GetSP());
}

namespace lldb_private {
namespace repro {

template <>
void RegisterMethods<SBAttachInfo>(Registry &R) {
  LLDB_REGISTER_CONSTRUCTOR(SBAttachInfo, ());
  LLDB_REGISTER_CONSTRUCTOR(SBAttachInfo, (lldb::pid_t));
  LLDB_REGISTER_CONSTRUCTOR(SBAttachInfo, (const char *, bool));
  LLDB_REGISTER_CONSTRUCTOR(SBAttachInfo, (const char *, bool, bool));
  LLDB_REGISTER_CONSTRUCTOR(SBAttachInfo, (const lldb::SBAttachInfo &));
  LLDB_REGISTER_METHOD(lldb::SBAttachInfo &,
                       SBAttachInfo, operator=,(const lldb::SBAttachInfo &));
  LLDB_REGISTER_METHOD(lldb::pid_t, SBAttachInfo, GetProcessID, ());
  LLDB_REGISTER_METHOD(void, SBAttachInfo, SetProcessID, (lldb::pid_t));
  LLDB_REGISTER_METHOD(uint32_t, SBAttachInfo, GetResumeCount, ());
  LLDB_REGISTER_METHOD(void, SBAttachInfo, SetResumeCount, (uint32_t));
  LLDB_REGISTER_METHOD(const char *, SBAttachInfo, GetProcessPluginName, ());
  LLDB_REGISTER_METHOD(void, SBAttachInfo, SetProcessPluginName,
                       (const char *));
  LLDB_REGISTER_METHOD(void, SBAttachInfo, SetExecutable, (const char *));
  LLDB_REGISTER_METHOD(void, SBAttachInfo, SetExecutable, (lldb::SBFileSpec));
  LLDB_REGISTER_METHOD(bool, SBAttachInfo, GetWaitForLaunch, ());
  LLDB_REGISTER_METHOD(void, SBAttachInfo, SetWaitForLaunch, (bool));
  LLDB_REGISTER_METHOD(void, SBAttachInfo, SetWaitForLaunch, (bool, bool));
  LLDB_REGISTER_METHOD(bool, SBAttachInfo, GetIgnoreExisting, ());
  LLDB_REGISTER_METHOD(void, SBAttachInfo, SetIgnoreExisting, (bool));
  LLDB_REGISTER_METHOD(uint32_t, SBAttachInfo, GetUserID, ());
  LLDB_REGISTER_METHOD(uint32_t, SBAttachInfo, GetGroupID, ());
  LLDB_REGISTER_METHOD(bool, SBAttachInfo, UserIDIsValid, ());
  LLDB_REGISTER_METHOD(bool, SBAttachInfo, GroupIDIsValid, ());
  LLDB_REGISTER_METHOD(void, SBAttachInfo, SetUserID, (uint32_t));
  LLDB_REGISTER_METHOD(void, SBAttachInfo, SetGroupID, (uint32_t));
  LLDB_REGISTER_METHOD(uint32_t, SBAttachInfo, GetEffectiveUserID, ());
  LLDB_REGISTER_METHOD(uint32_t, SBAttachInfo, GetEffectiveGroupID, ());
  LLDB_REGISTER_METHOD(bool, SBAttachInfo, EffectiveUserIDIsValid, ());
  LLDB_REGISTER_METHOD(bool, SBAttachInfo, EffectiveGroupIDIsValid, ());
  LLDB_REGISTER_METHOD(void, SBAttachInfo, SetEffectiveUserID, (uint32_t));
  LLDB_REGISTER_METHOD(void, SBAttachInfo, SetEffectiveGroupID, (uint32_t));
  LLDB_REGISTER_METHOD(lldb::pid_t, SBAttachInfo, GetParentProcessID, ());
  LLDB_REGISTER_METHOD(void, SBAttachInfo, SetParentProcessID, (lldb::pid_t));
  LLDB_REGISTER_METHOD(bool, SBAttachInfo, ParentProcessIDIsValid, ());
  LLDB_REGISTER_METHOD(lldb::SBListener, SBAttachInfo, GetListener, ());
  LLDB_REGISTER_METHOD(void, SBAttachInfo, SetListener, (lldb::SBListener &));
}

}
}
