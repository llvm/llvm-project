//===-- SBBreakpointName.cpp ----------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/API/SBBreakpointName.h"
#include "SBReproducerPrivate.h"
#include "lldb/API/SBDebugger.h"
#include "lldb/API/SBError.h"
#include "lldb/API/SBStream.h"
#include "lldb/API/SBStringList.h"
#include "lldb/API/SBTarget.h"

#include "lldb/Breakpoint/BreakpointName.h"
#include "lldb/Breakpoint/StoppointCallbackContext.h"
#include "lldb/Core/Debugger.h"
#include "lldb/Interpreter/CommandInterpreter.h"
#include "lldb/Interpreter/ScriptInterpreter.h"
#include "lldb/Target/Target.h"
#include "lldb/Target/ThreadSpec.h"
#include "lldb/Utility/Stream.h"

#include "SBBreakpointOptionCommon.h"

using namespace lldb;
using namespace lldb_private;

namespace lldb
{
class SBBreakpointNameImpl {
public:
  SBBreakpointNameImpl(TargetSP target_sp, const char *name) {
    if (!name || name[0] == '\0')
      return;
    m_name.assign(name);

    if (!target_sp)
      return;

    m_target_wp = target_sp;
  }

  SBBreakpointNameImpl(SBTarget &sb_target, const char *name);
  bool operator==(const SBBreakpointNameImpl &rhs);
  bool operator!=(const SBBreakpointNameImpl &rhs);

  // For now we take a simple approach and only keep the name, and relook up
  // the location when we need it.

  TargetSP GetTarget() const {
    return m_target_wp.lock();
  }

  const char *GetName() const {
    return m_name.c_str();
  }

  bool IsValid() const {
    return !m_name.empty() && m_target_wp.lock();
  }

  lldb_private::BreakpointName *GetBreakpointName() const;

private:
  TargetWP m_target_wp;
  std::string m_name;
};

SBBreakpointNameImpl::SBBreakpointNameImpl(SBTarget &sb_target,
                                           const char *name) {
  if (!name || name[0] == '\0')
    return;
  m_name.assign(name);

  if (!sb_target.IsValid())
    return;

  TargetSP target_sp = sb_target.GetSP();
  if (!target_sp)
    return;

  m_target_wp = target_sp;
}

bool SBBreakpointNameImpl::operator==(const SBBreakpointNameImpl &rhs) {
  return m_name == rhs.m_name && m_target_wp.lock() == rhs.m_target_wp.lock();
}

bool SBBreakpointNameImpl::operator!=(const SBBreakpointNameImpl &rhs) {
  return m_name != rhs.m_name || m_target_wp.lock() != rhs.m_target_wp.lock();
}

lldb_private::BreakpointName *SBBreakpointNameImpl::GetBreakpointName() const {
  if (!IsValid())
    return nullptr;
  TargetSP target_sp = GetTarget();
  if (!target_sp)
    return nullptr;
  Status error;
  return target_sp->FindBreakpointName(ConstString(m_name), true, error);
}

} // namespace lldb

SBBreakpointName::SBBreakpointName() {
  LLDB_RECORD_CONSTRUCTOR_NO_ARGS(SBBreakpointName);
}

SBBreakpointName::SBBreakpointName(SBTarget &sb_target, const char *name) {
  LLDB_RECORD_CONSTRUCTOR(SBBreakpointName, (lldb::SBTarget &, const char *),
                          sb_target, name);

  m_impl_up.reset(new SBBreakpointNameImpl(sb_target, name));
  // Call FindBreakpointName here to make sure the name is valid, reset if not:
  BreakpointName *bp_name = GetBreakpointName();
  if (!bp_name)
    m_impl_up.reset();
}

SBBreakpointName::SBBreakpointName(SBBreakpoint &sb_bkpt, const char *name) {
  LLDB_RECORD_CONSTRUCTOR(SBBreakpointName,
                          (lldb::SBBreakpoint &, const char *), sb_bkpt, name);

  if (!sb_bkpt.IsValid()) {
    m_impl_up.reset();
    return;
  }
  BreakpointSP bkpt_sp = sb_bkpt.GetSP();
  Target &target = bkpt_sp->GetTarget();

  m_impl_up.reset(new SBBreakpointNameImpl(target.shared_from_this(), name));

  // Call FindBreakpointName here to make sure the name is valid, reset if not:
  BreakpointName *bp_name = GetBreakpointName();
  if (!bp_name) {
    m_impl_up.reset();
    return;
  }

  // Now copy over the breakpoint's options:
  target.ConfigureBreakpointName(*bp_name, *bkpt_sp->GetOptions(),
                                 BreakpointName::Permissions());
}

SBBreakpointName::SBBreakpointName(const SBBreakpointName &rhs) {
  LLDB_RECORD_CONSTRUCTOR(SBBreakpointName, (const lldb::SBBreakpointName &),
                          rhs);

  if (!rhs.m_impl_up)
    return;
  else
    m_impl_up.reset(new SBBreakpointNameImpl(rhs.m_impl_up->GetTarget(),
                                             rhs.m_impl_up->GetName()));
}

SBBreakpointName::~SBBreakpointName() = default;

const SBBreakpointName &SBBreakpointName::
operator=(const SBBreakpointName &rhs) {
  LLDB_RECORD_METHOD(
      const lldb::SBBreakpointName &,
      SBBreakpointName, operator=,(const lldb::SBBreakpointName &), rhs);

  if (!rhs.m_impl_up) {
    m_impl_up.reset();
    return LLDB_RECORD_RESULT(*this);
  }

  m_impl_up.reset(new SBBreakpointNameImpl(rhs.m_impl_up->GetTarget(),
                                           rhs.m_impl_up->GetName()));
  return LLDB_RECORD_RESULT(*this);
}

bool SBBreakpointName::operator==(const lldb::SBBreakpointName &rhs) {
  LLDB_RECORD_METHOD(
      bool, SBBreakpointName, operator==,(const lldb::SBBreakpointName &), rhs);

  return *m_impl_up == *rhs.m_impl_up;
}

bool SBBreakpointName::operator!=(const lldb::SBBreakpointName &rhs) {
  LLDB_RECORD_METHOD(
      bool, SBBreakpointName, operator!=,(const lldb::SBBreakpointName &), rhs);

  return *m_impl_up != *rhs.m_impl_up;
}

bool SBBreakpointName::IsValid() const {
  LLDB_RECORD_METHOD_CONST_NO_ARGS(bool, SBBreakpointName, IsValid);
  return this->operator bool();
}
SBBreakpointName::operator bool() const {
  LLDB_RECORD_METHOD_CONST_NO_ARGS(bool, SBBreakpointName, operator bool);

  if (!m_impl_up)
    return false;
  return m_impl_up->IsValid();
}

const char *SBBreakpointName::GetName() const {
  LLDB_RECORD_METHOD_CONST_NO_ARGS(const char *, SBBreakpointName, GetName);

  if (!m_impl_up)
    return "<Invalid Breakpoint Name Object>";
  return m_impl_up->GetName();
}

void SBBreakpointName::SetEnabled(bool enable) {
  LLDB_RECORD_METHOD(void, SBBreakpointName, SetEnabled, (bool), enable);

  BreakpointName *bp_name = GetBreakpointName();
  if (!bp_name)
    return;

  std::lock_guard<std::recursive_mutex> guard(
        m_impl_up->GetTarget()->GetAPIMutex());

  bp_name->GetOptions().SetEnabled(enable);
}

void SBBreakpointName::UpdateName(BreakpointName &bp_name) {
  if (!IsValid())
    return;

  TargetSP target_sp = m_impl_up->GetTarget();
  if (!target_sp)
    return;
  target_sp->ApplyNameToBreakpoints(bp_name);

}

bool SBBreakpointName::IsEnabled() {
  LLDB_RECORD_METHOD_NO_ARGS(bool, SBBreakpointName, IsEnabled);

  BreakpointName *bp_name = GetBreakpointName();
  if (!bp_name)
    return false;

  std::lock_guard<std::recursive_mutex> guard(
        m_impl_up->GetTarget()->GetAPIMutex());

  return bp_name->GetOptions().IsEnabled();
}

void SBBreakpointName::SetOneShot(bool one_shot) {
  LLDB_RECORD_METHOD(void, SBBreakpointName, SetOneShot, (bool), one_shot);

  BreakpointName *bp_name = GetBreakpointName();
  if (!bp_name)
    return;

  std::lock_guard<std::recursive_mutex> guard(
        m_impl_up->GetTarget()->GetAPIMutex());

  bp_name->GetOptions().SetOneShot(one_shot);
  UpdateName(*bp_name);
}

bool SBBreakpointName::IsOneShot() const {
  LLDB_RECORD_METHOD_CONST_NO_ARGS(bool, SBBreakpointName, IsOneShot);

  const BreakpointName *bp_name = GetBreakpointName();
  if (!bp_name)
    return false;

  std::lock_guard<std::recursive_mutex> guard(
        m_impl_up->GetTarget()->GetAPIMutex());

  return bp_name->GetOptions().IsOneShot();
}

void SBBreakpointName::SetIgnoreCount(uint32_t count) {
  LLDB_RECORD_METHOD(void, SBBreakpointName, SetIgnoreCount, (uint32_t), count);

  BreakpointName *bp_name = GetBreakpointName();
  if (!bp_name)
    return;

  std::lock_guard<std::recursive_mutex> guard(
        m_impl_up->GetTarget()->GetAPIMutex());

  bp_name->GetOptions().SetIgnoreCount(count);
  UpdateName(*bp_name);
}

uint32_t SBBreakpointName::GetIgnoreCount() const {
  LLDB_RECORD_METHOD_CONST_NO_ARGS(uint32_t, SBBreakpointName, GetIgnoreCount);

  BreakpointName *bp_name = GetBreakpointName();
  if (!bp_name)
    return false;

  std::lock_guard<std::recursive_mutex> guard(
        m_impl_up->GetTarget()->GetAPIMutex());

  return bp_name->GetOptions().GetIgnoreCount();
}

void SBBreakpointName::SetCondition(const char *condition) {
  LLDB_RECORD_METHOD(void, SBBreakpointName, SetCondition, (const char *),
                     condition);

  BreakpointName *bp_name = GetBreakpointName();
  if (!bp_name)
    return;

  std::lock_guard<std::recursive_mutex> guard(
        m_impl_up->GetTarget()->GetAPIMutex());

  bp_name->GetOptions().SetCondition(condition);
  UpdateName(*bp_name);
}

const char *SBBreakpointName::GetCondition() {
  LLDB_RECORD_METHOD_NO_ARGS(const char *, SBBreakpointName, GetCondition);

  BreakpointName *bp_name = GetBreakpointName();
  if (!bp_name)
    return nullptr;

  std::lock_guard<std::recursive_mutex> guard(
        m_impl_up->GetTarget()->GetAPIMutex());

  return bp_name->GetOptions().GetConditionText();
}

void SBBreakpointName::SetAutoContinue(bool auto_continue) {
  LLDB_RECORD_METHOD(void, SBBreakpointName, SetAutoContinue, (bool),
                     auto_continue);

  BreakpointName *bp_name = GetBreakpointName();
  if (!bp_name)
    return;

  std::lock_guard<std::recursive_mutex> guard(
        m_impl_up->GetTarget()->GetAPIMutex());

  bp_name->GetOptions().SetAutoContinue(auto_continue);
  UpdateName(*bp_name);
}

bool SBBreakpointName::GetAutoContinue() {
  LLDB_RECORD_METHOD_NO_ARGS(bool, SBBreakpointName, GetAutoContinue);

  BreakpointName *bp_name = GetBreakpointName();
  if (!bp_name)
    return false;

  std::lock_guard<std::recursive_mutex> guard(
        m_impl_up->GetTarget()->GetAPIMutex());

  return bp_name->GetOptions().IsAutoContinue();
}

void SBBreakpointName::SetThreadID(tid_t tid) {
  LLDB_RECORD_METHOD(void, SBBreakpointName, SetThreadID, (lldb::tid_t), tid);

  BreakpointName *bp_name = GetBreakpointName();
  if (!bp_name)
    return;

  std::lock_guard<std::recursive_mutex> guard(
        m_impl_up->GetTarget()->GetAPIMutex());

  bp_name->GetOptions().SetThreadID(tid);
  UpdateName(*bp_name);
}

tid_t SBBreakpointName::GetThreadID() {
  LLDB_RECORD_METHOD_NO_ARGS(lldb::tid_t, SBBreakpointName, GetThreadID);

  BreakpointName *bp_name = GetBreakpointName();
  if (!bp_name)
    return LLDB_INVALID_THREAD_ID;

  std::lock_guard<std::recursive_mutex> guard(
        m_impl_up->GetTarget()->GetAPIMutex());

  return bp_name->GetOptions().GetThreadSpec()->GetTID();
}

void SBBreakpointName::SetThreadIndex(uint32_t index) {
  LLDB_RECORD_METHOD(void, SBBreakpointName, SetThreadIndex, (uint32_t), index);

  BreakpointName *bp_name = GetBreakpointName();
  if (!bp_name)
    return;

  std::lock_guard<std::recursive_mutex> guard(
        m_impl_up->GetTarget()->GetAPIMutex());

  bp_name->GetOptions().GetThreadSpec()->SetIndex(index);
  UpdateName(*bp_name);
}

uint32_t SBBreakpointName::GetThreadIndex() const {
  LLDB_RECORD_METHOD_CONST_NO_ARGS(uint32_t, SBBreakpointName, GetThreadIndex);

  BreakpointName *bp_name = GetBreakpointName();
  if (!bp_name)
    return LLDB_INVALID_THREAD_ID;

  std::lock_guard<std::recursive_mutex> guard(
        m_impl_up->GetTarget()->GetAPIMutex());

  return bp_name->GetOptions().GetThreadSpec()->GetIndex();
}

void SBBreakpointName::SetThreadName(const char *thread_name) {
  LLDB_RECORD_METHOD(void, SBBreakpointName, SetThreadName, (const char *),
                     thread_name);

  BreakpointName *bp_name = GetBreakpointName();
  if (!bp_name)
    return;

  std::lock_guard<std::recursive_mutex> guard(
        m_impl_up->GetTarget()->GetAPIMutex());

  bp_name->GetOptions().GetThreadSpec()->SetName(thread_name);
  UpdateName(*bp_name);
}

const char *SBBreakpointName::GetThreadName() const {
  LLDB_RECORD_METHOD_CONST_NO_ARGS(const char *, SBBreakpointName,
                                   GetThreadName);

  BreakpointName *bp_name = GetBreakpointName();
  if (!bp_name)
    return nullptr;

  std::lock_guard<std::recursive_mutex> guard(
        m_impl_up->GetTarget()->GetAPIMutex());

  return bp_name->GetOptions().GetThreadSpec()->GetName();
}

void SBBreakpointName::SetQueueName(const char *queue_name) {
  LLDB_RECORD_METHOD(void, SBBreakpointName, SetQueueName, (const char *),
                     queue_name);

  BreakpointName *bp_name = GetBreakpointName();
  if (!bp_name)
    return;

  std::lock_guard<std::recursive_mutex> guard(
        m_impl_up->GetTarget()->GetAPIMutex());

  bp_name->GetOptions().GetThreadSpec()->SetQueueName(queue_name);
  UpdateName(*bp_name);
}

const char *SBBreakpointName::GetQueueName() const {
  LLDB_RECORD_METHOD_CONST_NO_ARGS(const char *, SBBreakpointName,
                                   GetQueueName);

  BreakpointName *bp_name = GetBreakpointName();
  if (!bp_name)
    return nullptr;

  std::lock_guard<std::recursive_mutex> guard(
        m_impl_up->GetTarget()->GetAPIMutex());

  return bp_name->GetOptions().GetThreadSpec()->GetQueueName();
}

void SBBreakpointName::SetCommandLineCommands(SBStringList &commands) {
  LLDB_RECORD_METHOD(void, SBBreakpointName, SetCommandLineCommands,
                     (lldb::SBStringList &), commands);

  BreakpointName *bp_name = GetBreakpointName();
  if (!bp_name)
    return;
  if (commands.GetSize() == 0)
    return;


  std::lock_guard<std::recursive_mutex> guard(
        m_impl_up->GetTarget()->GetAPIMutex());
  std::unique_ptr<BreakpointOptions::CommandData> cmd_data_up(
      new BreakpointOptions::CommandData(*commands, eScriptLanguageNone));

  bp_name->GetOptions().SetCommandDataCallback(cmd_data_up);
  UpdateName(*bp_name);
}

bool SBBreakpointName::GetCommandLineCommands(SBStringList &commands) {
  LLDB_RECORD_METHOD(bool, SBBreakpointName, GetCommandLineCommands,
                     (lldb::SBStringList &), commands);

  BreakpointName *bp_name = GetBreakpointName();
  if (!bp_name)
    return false;

  StringList command_list;
  bool has_commands =
      bp_name->GetOptions().GetCommandLineCallbacks(command_list);
  if (has_commands)
    commands.AppendList(command_list);
  return has_commands;
}

const char *SBBreakpointName::GetHelpString() const {
  LLDB_RECORD_METHOD_CONST_NO_ARGS(const char *, SBBreakpointName,
                                   GetHelpString);

  BreakpointName *bp_name = GetBreakpointName();
  if (!bp_name)
    return "";

  return bp_name->GetHelp();
}

void SBBreakpointName::SetHelpString(const char *help_string) {
  LLDB_RECORD_METHOD(void, SBBreakpointName, SetHelpString, (const char *),
                     help_string);

  BreakpointName *bp_name = GetBreakpointName();
  if (!bp_name)
    return;


  std::lock_guard<std::recursive_mutex> guard(
        m_impl_up->GetTarget()->GetAPIMutex());
  bp_name->SetHelp(help_string);
}

bool SBBreakpointName::GetDescription(SBStream &s) {
  LLDB_RECORD_METHOD(bool, SBBreakpointName, GetDescription, (lldb::SBStream &),
                     s);

  BreakpointName *bp_name = GetBreakpointName();
  if (!bp_name)
  {
    s.Printf("No value");
    return false;
  }

  std::lock_guard<std::recursive_mutex> guard(
        m_impl_up->GetTarget()->GetAPIMutex());
  bp_name->GetDescription(s.get(), eDescriptionLevelFull);
  return true;
}

void SBBreakpointName::SetCallback(SBBreakpointHitCallback callback,
                                   void *baton) {
  LLDB_RECORD_DUMMY(void, SBBreakpointName, SetCallback,
                    (lldb::SBBreakpointHitCallback, void *), callback, baton);

  BreakpointName *bp_name = GetBreakpointName();
  if (!bp_name)
    return;
  std::lock_guard<std::recursive_mutex> guard(
        m_impl_up->GetTarget()->GetAPIMutex());

  BatonSP baton_sp(new SBBreakpointCallbackBaton(callback, baton));
  bp_name->GetOptions().SetCallback(SBBreakpointCallbackBaton
                                       ::PrivateBreakpointHitCallback,
                                    baton_sp,
                                    false);
  UpdateName(*bp_name);
}

void SBBreakpointName::SetScriptCallbackFunction(
    const char *callback_function_name) {
  LLDB_RECORD_METHOD(void, SBBreakpointName, SetScriptCallbackFunction,
                     (const char *), callback_function_name);

  BreakpointName *bp_name = GetBreakpointName();
  if (!bp_name)
    return;

  std::lock_guard<std::recursive_mutex> guard(
        m_impl_up->GetTarget()->GetAPIMutex());

  BreakpointOptions &bp_options = bp_name->GetOptions();
  m_impl_up->GetTarget()
      ->GetDebugger()
      .GetScriptInterpreter()
      ->SetBreakpointCommandCallbackFunction(&bp_options,
                                             callback_function_name);
  UpdateName(*bp_name);
}

SBError
SBBreakpointName::SetScriptCallbackBody(const char *callback_body_text) {
  LLDB_RECORD_METHOD(lldb::SBError, SBBreakpointName, SetScriptCallbackBody,
                     (const char *), callback_body_text);

  SBError sb_error;
  BreakpointName *bp_name = GetBreakpointName();
  if (!bp_name)
    return LLDB_RECORD_RESULT(sb_error);

  std::lock_guard<std::recursive_mutex> guard(
        m_impl_up->GetTarget()->GetAPIMutex());

  BreakpointOptions &bp_options = bp_name->GetOptions();
  Status error =
      m_impl_up->GetTarget()
          ->GetDebugger()
          .GetScriptInterpreter()
          ->SetBreakpointCommandCallback(&bp_options, callback_body_text);
  sb_error.SetError(error);
  if (!sb_error.Fail())
    UpdateName(*bp_name);

  return LLDB_RECORD_RESULT(sb_error);
}

bool SBBreakpointName::GetAllowList() const {
  LLDB_RECORD_METHOD_CONST_NO_ARGS(bool, SBBreakpointName, GetAllowList);

  BreakpointName *bp_name = GetBreakpointName();
  if (!bp_name)
    return false;
  return bp_name->GetPermissions().GetAllowList();
}

void SBBreakpointName::SetAllowList(bool value) {
  LLDB_RECORD_METHOD(void, SBBreakpointName, SetAllowList, (bool), value);


  BreakpointName *bp_name = GetBreakpointName();
  if (!bp_name)
    return;
  bp_name->GetPermissions().SetAllowList(value);
}

bool SBBreakpointName::GetAllowDelete() {
  LLDB_RECORD_METHOD_NO_ARGS(bool, SBBreakpointName, GetAllowDelete);

  BreakpointName *bp_name = GetBreakpointName();
  if (!bp_name)
    return false;
  return bp_name->GetPermissions().GetAllowDelete();
}

void SBBreakpointName::SetAllowDelete(bool value) {
  LLDB_RECORD_METHOD(void, SBBreakpointName, SetAllowDelete, (bool), value);


  BreakpointName *bp_name = GetBreakpointName();
  if (!bp_name)
    return;
  bp_name->GetPermissions().SetAllowDelete(value);
}

bool SBBreakpointName::GetAllowDisable() {
  LLDB_RECORD_METHOD_NO_ARGS(bool, SBBreakpointName, GetAllowDisable);

  BreakpointName *bp_name = GetBreakpointName();
  if (!bp_name)
    return false;
  return bp_name->GetPermissions().GetAllowDisable();
}

void SBBreakpointName::SetAllowDisable(bool value) {
  LLDB_RECORD_METHOD(void, SBBreakpointName, SetAllowDisable, (bool), value);

  BreakpointName *bp_name = GetBreakpointName();
  if (!bp_name)
    return;
  bp_name->GetPermissions().SetAllowDisable(value);
}

lldb_private::BreakpointName *SBBreakpointName::GetBreakpointName() const
{
  if (!IsValid())
    return nullptr;
  return m_impl_up->GetBreakpointName();
}


namespace lldb_private {
namespace repro {

template <>
void RegisterMethods<SBBreakpointName>(Registry &R) {
  LLDB_REGISTER_CONSTRUCTOR(SBBreakpointName, ());
  LLDB_REGISTER_CONSTRUCTOR(SBBreakpointName,
                            (lldb::SBTarget &, const char *));
  LLDB_REGISTER_CONSTRUCTOR(SBBreakpointName,
                            (lldb::SBBreakpoint &, const char *));
  LLDB_REGISTER_CONSTRUCTOR(SBBreakpointName,
                            (const lldb::SBBreakpointName &));
  LLDB_REGISTER_METHOD(
      const lldb::SBBreakpointName &,
      SBBreakpointName, operator=,(const lldb::SBBreakpointName &));
  LLDB_REGISTER_METHOD(
      bool, SBBreakpointName, operator==,(const lldb::SBBreakpointName &));
  LLDB_REGISTER_METHOD(
      bool, SBBreakpointName, operator!=,(const lldb::SBBreakpointName &));
  LLDB_REGISTER_METHOD_CONST(bool, SBBreakpointName, IsValid, ());
  LLDB_REGISTER_METHOD_CONST(bool, SBBreakpointName, operator bool, ());
  LLDB_REGISTER_METHOD_CONST(const char *, SBBreakpointName, GetName, ());
  LLDB_REGISTER_METHOD(void, SBBreakpointName, SetEnabled, (bool));
  LLDB_REGISTER_METHOD(bool, SBBreakpointName, IsEnabled, ());
  LLDB_REGISTER_METHOD(void, SBBreakpointName, SetOneShot, (bool));
  LLDB_REGISTER_METHOD_CONST(bool, SBBreakpointName, IsOneShot, ());
  LLDB_REGISTER_METHOD(void, SBBreakpointName, SetIgnoreCount, (uint32_t));
  LLDB_REGISTER_METHOD_CONST(uint32_t, SBBreakpointName, GetIgnoreCount, ());
  LLDB_REGISTER_METHOD(void, SBBreakpointName, SetCondition, (const char *));
  LLDB_REGISTER_METHOD(const char *, SBBreakpointName, GetCondition, ());
  LLDB_REGISTER_METHOD(void, SBBreakpointName, SetAutoContinue, (bool));
  LLDB_REGISTER_METHOD(bool, SBBreakpointName, GetAutoContinue, ());
  LLDB_REGISTER_METHOD(void, SBBreakpointName, SetThreadID, (lldb::tid_t));
  LLDB_REGISTER_METHOD(lldb::tid_t, SBBreakpointName, GetThreadID, ());
  LLDB_REGISTER_METHOD(void, SBBreakpointName, SetThreadIndex, (uint32_t));
  LLDB_REGISTER_METHOD_CONST(uint32_t, SBBreakpointName, GetThreadIndex, ());
  LLDB_REGISTER_METHOD(void, SBBreakpointName, SetThreadName, (const char *));
  LLDB_REGISTER_METHOD_CONST(const char *, SBBreakpointName, GetThreadName,
                             ());
  LLDB_REGISTER_METHOD(void, SBBreakpointName, SetQueueName, (const char *));
  LLDB_REGISTER_METHOD_CONST(const char *, SBBreakpointName, GetQueueName,
                             ());
  LLDB_REGISTER_METHOD(void, SBBreakpointName, SetCommandLineCommands,
                       (lldb::SBStringList &));
  LLDB_REGISTER_METHOD(bool, SBBreakpointName, GetCommandLineCommands,
                       (lldb::SBStringList &));
  LLDB_REGISTER_METHOD_CONST(const char *, SBBreakpointName, GetHelpString,
                             ());
  LLDB_REGISTER_METHOD(void, SBBreakpointName, SetHelpString, (const char *));
  LLDB_REGISTER_METHOD(bool, SBBreakpointName, GetDescription,
                       (lldb::SBStream &));
  LLDB_REGISTER_METHOD(void, SBBreakpointName, SetScriptCallbackFunction,
                       (const char *));
  LLDB_REGISTER_METHOD(lldb::SBError, SBBreakpointName, SetScriptCallbackBody,
                       (const char *));
  LLDB_REGISTER_METHOD_CONST(bool, SBBreakpointName, GetAllowList, ());
  LLDB_REGISTER_METHOD(void, SBBreakpointName, SetAllowList, (bool));
  LLDB_REGISTER_METHOD(bool, SBBreakpointName, GetAllowDelete, ());
  LLDB_REGISTER_METHOD(void, SBBreakpointName, SetAllowDelete, (bool));
  LLDB_REGISTER_METHOD(bool, SBBreakpointName, GetAllowDisable, ());
  LLDB_REGISTER_METHOD(void, SBBreakpointName, SetAllowDisable, (bool));
}

}
}
