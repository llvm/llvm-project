//===-- SBProcessInfo.cpp ---------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/API/SBProcessInfo.h"

#include "lldb/API/SBFileSpec.h"
#include "lldb/Target/Process.h"

using namespace lldb;
using namespace lldb_private;

SBProcessInfo::SBProcessInfo() : m_opaque_up() {}

SBProcessInfo::SBProcessInfo(const SBProcessInfo &rhs) : m_opaque_up() {
  if (rhs.IsValid()) {
    ref() = *rhs.m_opaque_up;
  }
}

SBProcessInfo::~SBProcessInfo() {}

SBProcessInfo &SBProcessInfo::operator=(const SBProcessInfo &rhs) {
  if (this != &rhs) {
    if (rhs.IsValid())
      ref() = *rhs.m_opaque_up;
    else
      m_opaque_up.reset();
  }
  return *this;
}

ProcessInstanceInfo &SBProcessInfo::ref() {
  if (m_opaque_up == nullptr) {
    m_opaque_up.reset(new ProcessInstanceInfo());
  }
  return *m_opaque_up;
}

void SBProcessInfo::SetProcessInfo(const ProcessInstanceInfo &proc_info_ref) {
  ref() = proc_info_ref;
}

bool SBProcessInfo::IsValid() const { return m_opaque_up != nullptr; }

const char *SBProcessInfo::GetName() {
  const char *name = nullptr;
  if (m_opaque_up) {
    name = m_opaque_up->GetName();
  }
  return name;
}

SBFileSpec SBProcessInfo::GetExecutableFile() {
  SBFileSpec file_spec;
  if (m_opaque_up) {
    file_spec.SetFileSpec(m_opaque_up->GetExecutableFile());
  }
  return file_spec;
}

lldb::pid_t SBProcessInfo::GetProcessID() {
  lldb::pid_t proc_id = LLDB_INVALID_PROCESS_ID;
  if (m_opaque_up) {
    proc_id = m_opaque_up->GetProcessID();
  }
  return proc_id;
}

uint32_t SBProcessInfo::GetUserID() {
  uint32_t user_id = UINT32_MAX;
  if (m_opaque_up) {
    user_id = m_opaque_up->GetUserID();
  }
  return user_id;
}

uint32_t SBProcessInfo::GetGroupID() {
  uint32_t group_id = UINT32_MAX;
  if (m_opaque_up) {
    group_id = m_opaque_up->GetGroupID();
  }
  return group_id;
}

bool SBProcessInfo::UserIDIsValid() {
  bool is_valid = false;
  if (m_opaque_up) {
    is_valid = m_opaque_up->UserIDIsValid();
  }
  return is_valid;
}

bool SBProcessInfo::GroupIDIsValid() {
  bool is_valid = false;
  if (m_opaque_up) {
    is_valid = m_opaque_up->GroupIDIsValid();
  }
  return is_valid;
}

uint32_t SBProcessInfo::GetEffectiveUserID() {
  uint32_t user_id = UINT32_MAX;
  if (m_opaque_up) {
    user_id = m_opaque_up->GetEffectiveUserID();
  }
  return user_id;
}

uint32_t SBProcessInfo::GetEffectiveGroupID() {
  uint32_t group_id = UINT32_MAX;
  if (m_opaque_up) {
    group_id = m_opaque_up->GetEffectiveGroupID();
  }
  return group_id;
}

bool SBProcessInfo::EffectiveUserIDIsValid() {
  bool is_valid = false;
  if (m_opaque_up) {
    is_valid = m_opaque_up->EffectiveUserIDIsValid();
  }
  return is_valid;
}

bool SBProcessInfo::EffectiveGroupIDIsValid() {
  bool is_valid = false;
  if (m_opaque_up) {
    is_valid = m_opaque_up->EffectiveGroupIDIsValid();
  }
  return is_valid;
}

lldb::pid_t SBProcessInfo::GetParentProcessID() {
  lldb::pid_t proc_id = LLDB_INVALID_PROCESS_ID;
  if (m_opaque_up) {
    proc_id = m_opaque_up->GetParentProcessID();
  }
  return proc_id;
}
