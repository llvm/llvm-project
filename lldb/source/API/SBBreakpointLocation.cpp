//===-- SBBreakpointLocation.cpp --------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/API/SBBreakpointLocation.h"
#include "SBReproducerPrivate.h"
#include "lldb/API/SBAddress.h"
#include "lldb/API/SBDebugger.h"
#include "lldb/API/SBDefines.h"
#include "lldb/API/SBStream.h"
#include "lldb/API/SBStringList.h"

#include "lldb/Breakpoint/Breakpoint.h"
#include "lldb/Breakpoint/BreakpointLocation.h"
#include "lldb/Core/Debugger.h"
#include "lldb/Core/StreamFile.h"
#include "lldb/Interpreter/CommandInterpreter.h"
#include "lldb/Interpreter/ScriptInterpreter.h"
#include "lldb/Target/Target.h"
#include "lldb/Target/ThreadSpec.h"
#include "lldb/Utility/Log.h"
#include "lldb/Utility/Stream.h"
#include "lldb/lldb-defines.h"
#include "lldb/lldb-types.h"

using namespace lldb;
using namespace lldb_private;

SBBreakpointLocation::SBBreakpointLocation() {
  LLDB_RECORD_CONSTRUCTOR_NO_ARGS(SBBreakpointLocation);
}

SBBreakpointLocation::SBBreakpointLocation(
    const lldb::BreakpointLocationSP &break_loc_sp)
    : m_opaque_wp(break_loc_sp) {
  LLDB_RECORD_CONSTRUCTOR(SBBreakpointLocation,
                          (const lldb::BreakpointLocationSP &), break_loc_sp);

  Log *log(lldb_private::GetLogIfAllCategoriesSet(LIBLLDB_LOG_API));

  if (log) {
    SBStream sstr;
    GetDescription(sstr, lldb::eDescriptionLevelBrief);
    LLDB_LOG(log, "location = {0} ({1})", break_loc_sp.get(), sstr.GetData());
  }
}

SBBreakpointLocation::SBBreakpointLocation(const SBBreakpointLocation &rhs)
    : m_opaque_wp(rhs.m_opaque_wp) {
  LLDB_RECORD_CONSTRUCTOR(SBBreakpointLocation,
                          (const lldb::SBBreakpointLocation &), rhs);
}

const SBBreakpointLocation &SBBreakpointLocation::
operator=(const SBBreakpointLocation &rhs) {
  LLDB_RECORD_METHOD(
      const lldb::SBBreakpointLocation &,
      SBBreakpointLocation, operator=,(const lldb::SBBreakpointLocation &),
      rhs);

  m_opaque_wp = rhs.m_opaque_wp;
  return *this;
}

SBBreakpointLocation::~SBBreakpointLocation() {}

BreakpointLocationSP SBBreakpointLocation::GetSP() const {
  return m_opaque_wp.lock();
}

bool SBBreakpointLocation::IsValid() const {
  LLDB_RECORD_METHOD_CONST_NO_ARGS(bool, SBBreakpointLocation, IsValid);

  return bool(GetSP());
}

SBAddress SBBreakpointLocation::GetAddress() {
  LLDB_RECORD_METHOD_NO_ARGS(lldb::SBAddress, SBBreakpointLocation, GetAddress);

  BreakpointLocationSP loc_sp = GetSP();
  if (loc_sp) {
    return LLDB_RECORD_RESULT(SBAddress(&loc_sp->GetAddress()));
  }

  return LLDB_RECORD_RESULT(SBAddress());
}

addr_t SBBreakpointLocation::GetLoadAddress() {
  LLDB_RECORD_METHOD_NO_ARGS(lldb::addr_t, SBBreakpointLocation,
                             GetLoadAddress);

  addr_t ret_addr = LLDB_INVALID_ADDRESS;
  BreakpointLocationSP loc_sp = GetSP();

  if (loc_sp) {
    std::lock_guard<std::recursive_mutex> guard(
        loc_sp->GetTarget().GetAPIMutex());
    ret_addr = loc_sp->GetLoadAddress();
  }

  return ret_addr;
}

void SBBreakpointLocation::SetEnabled(bool enabled) {
  LLDB_RECORD_METHOD(void, SBBreakpointLocation, SetEnabled, (bool), enabled);

  BreakpointLocationSP loc_sp = GetSP();
  if (loc_sp) {
    std::lock_guard<std::recursive_mutex> guard(
        loc_sp->GetTarget().GetAPIMutex());
    loc_sp->SetEnabled(enabled);
  }
}

bool SBBreakpointLocation::IsEnabled() {
  LLDB_RECORD_METHOD_NO_ARGS(bool, SBBreakpointLocation, IsEnabled);

  BreakpointLocationSP loc_sp = GetSP();
  if (loc_sp) {
    std::lock_guard<std::recursive_mutex> guard(
        loc_sp->GetTarget().GetAPIMutex());
    return loc_sp->IsEnabled();
  } else
    return false;
}

uint32_t SBBreakpointLocation::GetHitCount() {
  LLDB_RECORD_METHOD_NO_ARGS(uint32_t, SBBreakpointLocation, GetHitCount);

  BreakpointLocationSP loc_sp = GetSP();
  if (loc_sp) {
    std::lock_guard<std::recursive_mutex> guard(
        loc_sp->GetTarget().GetAPIMutex());
    return loc_sp->GetHitCount();
  } else
    return 0;
}

uint32_t SBBreakpointLocation::GetIgnoreCount() {
  LLDB_RECORD_METHOD_NO_ARGS(uint32_t, SBBreakpointLocation, GetIgnoreCount);

  BreakpointLocationSP loc_sp = GetSP();
  if (loc_sp) {
    std::lock_guard<std::recursive_mutex> guard(
        loc_sp->GetTarget().GetAPIMutex());
    return loc_sp->GetIgnoreCount();
  } else
    return 0;
}

void SBBreakpointLocation::SetIgnoreCount(uint32_t n) {
  LLDB_RECORD_METHOD(void, SBBreakpointLocation, SetIgnoreCount, (uint32_t), n);

  BreakpointLocationSP loc_sp = GetSP();
  if (loc_sp) {
    std::lock_guard<std::recursive_mutex> guard(
        loc_sp->GetTarget().GetAPIMutex());
    loc_sp->SetIgnoreCount(n);
  }
}

void SBBreakpointLocation::SetCondition(const char *condition) {
  LLDB_RECORD_METHOD(void, SBBreakpointLocation, SetCondition, (const char *),
                     condition);

  BreakpointLocationSP loc_sp = GetSP();
  if (loc_sp) {
    std::lock_guard<std::recursive_mutex> guard(
        loc_sp->GetTarget().GetAPIMutex());
    loc_sp->SetCondition(condition);
  }
}

const char *SBBreakpointLocation::GetCondition() {
  LLDB_RECORD_METHOD_NO_ARGS(const char *, SBBreakpointLocation, GetCondition);

  BreakpointLocationSP loc_sp = GetSP();
  if (loc_sp) {
    std::lock_guard<std::recursive_mutex> guard(
        loc_sp->GetTarget().GetAPIMutex());
    return loc_sp->GetConditionText();
  }
  return NULL;
}

void SBBreakpointLocation::SetAutoContinue(bool auto_continue) {
  LLDB_RECORD_METHOD(void, SBBreakpointLocation, SetAutoContinue, (bool),
                     auto_continue);

  BreakpointLocationSP loc_sp = GetSP();
  if (loc_sp) {
    std::lock_guard<std::recursive_mutex> guard(
        loc_sp->GetTarget().GetAPIMutex());
    loc_sp->SetAutoContinue(auto_continue);
  }
}

bool SBBreakpointLocation::GetAutoContinue() {
  LLDB_RECORD_METHOD_NO_ARGS(bool, SBBreakpointLocation, GetAutoContinue);

  BreakpointLocationSP loc_sp = GetSP();
  if (loc_sp) {
    std::lock_guard<std::recursive_mutex> guard(
        loc_sp->GetTarget().GetAPIMutex());
    return loc_sp->IsAutoContinue();
  }
  return false;
}

void SBBreakpointLocation::SetScriptCallbackFunction(
    const char *callback_function_name) {
  LLDB_RECORD_METHOD(void, SBBreakpointLocation, SetScriptCallbackFunction,
                     (const char *), callback_function_name);

  Log *log(lldb_private::GetLogIfAllCategoriesSet(LIBLLDB_LOG_API));
  BreakpointLocationSP loc_sp = GetSP();
  LLDB_LOG(log, "location = {0}, callback = {1}", loc_sp.get(),
           callback_function_name);

  if (loc_sp) {
    std::lock_guard<std::recursive_mutex> guard(
        loc_sp->GetTarget().GetAPIMutex());
    BreakpointOptions *bp_options = loc_sp->GetLocationOptions();
    loc_sp->GetBreakpoint()
        .GetTarget()
        .GetDebugger()
        .GetCommandInterpreter()
        .GetScriptInterpreter()
        ->SetBreakpointCommandCallbackFunction(bp_options,
                                               callback_function_name);
  }
}

SBError
SBBreakpointLocation::SetScriptCallbackBody(const char *callback_body_text) {
  LLDB_RECORD_METHOD(lldb::SBError, SBBreakpointLocation, SetScriptCallbackBody,
                     (const char *), callback_body_text);

  Log *log(lldb_private::GetLogIfAllCategoriesSet(LIBLLDB_LOG_API));
  BreakpointLocationSP loc_sp = GetSP();
  LLDB_LOG(log, "location = {0}: callback body:\n{1}", loc_sp.get(),
           callback_body_text);

  SBError sb_error;
  if (loc_sp) {
    std::lock_guard<std::recursive_mutex> guard(
        loc_sp->GetTarget().GetAPIMutex());
    BreakpointOptions *bp_options = loc_sp->GetLocationOptions();
    Status error =
        loc_sp->GetBreakpoint()
            .GetTarget()
            .GetDebugger()
            .GetCommandInterpreter()
            .GetScriptInterpreter()
            ->SetBreakpointCommandCallback(bp_options, callback_body_text);
    sb_error.SetError(error);
  } else
    sb_error.SetErrorString("invalid breakpoint");

  return LLDB_RECORD_RESULT(sb_error);
}

void SBBreakpointLocation::SetCommandLineCommands(SBStringList &commands) {
  LLDB_RECORD_METHOD(void, SBBreakpointLocation, SetCommandLineCommands,
                     (lldb::SBStringList &), commands);

  BreakpointLocationSP loc_sp = GetSP();
  if (!loc_sp)
    return;
  if (commands.GetSize() == 0)
    return;

  std::lock_guard<std::recursive_mutex> guard(
      loc_sp->GetTarget().GetAPIMutex());
  std::unique_ptr<BreakpointOptions::CommandData> cmd_data_up(
      new BreakpointOptions::CommandData(*commands, eScriptLanguageNone));

  loc_sp->GetLocationOptions()->SetCommandDataCallback(cmd_data_up);
}

bool SBBreakpointLocation::GetCommandLineCommands(SBStringList &commands) {
  LLDB_RECORD_METHOD(bool, SBBreakpointLocation, GetCommandLineCommands,
                     (lldb::SBStringList &), commands);

  BreakpointLocationSP loc_sp = GetSP();
  if (!loc_sp)
    return false;
  StringList command_list;
  bool has_commands =
      loc_sp->GetLocationOptions()->GetCommandLineCallbacks(command_list);
  if (has_commands)
    commands.AppendList(command_list);
  return has_commands;
}

void SBBreakpointLocation::SetThreadID(tid_t thread_id) {
  LLDB_RECORD_METHOD(void, SBBreakpointLocation, SetThreadID, (lldb::tid_t),
                     thread_id);

  BreakpointLocationSP loc_sp = GetSP();
  if (loc_sp) {
    std::lock_guard<std::recursive_mutex> guard(
        loc_sp->GetTarget().GetAPIMutex());
    loc_sp->SetThreadID(thread_id);
  }
}

tid_t SBBreakpointLocation::GetThreadID() {
  LLDB_RECORD_METHOD_NO_ARGS(lldb::tid_t, SBBreakpointLocation, GetThreadID);

  tid_t tid = LLDB_INVALID_THREAD_ID;
  BreakpointLocationSP loc_sp = GetSP();
  if (loc_sp) {
    std::lock_guard<std::recursive_mutex> guard(
        loc_sp->GetTarget().GetAPIMutex());
    return loc_sp->GetThreadID();
  }
  return tid;
}

void SBBreakpointLocation::SetThreadIndex(uint32_t index) {
  LLDB_RECORD_METHOD(void, SBBreakpointLocation, SetThreadIndex, (uint32_t),
                     index);

  BreakpointLocationSP loc_sp = GetSP();
  if (loc_sp) {
    std::lock_guard<std::recursive_mutex> guard(
        loc_sp->GetTarget().GetAPIMutex());
    loc_sp->SetThreadIndex(index);
  }
}

uint32_t SBBreakpointLocation::GetThreadIndex() const {
  LLDB_RECORD_METHOD_CONST_NO_ARGS(uint32_t, SBBreakpointLocation,
                                   GetThreadIndex);

  uint32_t thread_idx = UINT32_MAX;
  BreakpointLocationSP loc_sp = GetSP();
  if (loc_sp) {
    std::lock_guard<std::recursive_mutex> guard(
        loc_sp->GetTarget().GetAPIMutex());
    return loc_sp->GetThreadIndex();
  }
  return thread_idx;
}

void SBBreakpointLocation::SetThreadName(const char *thread_name) {
  LLDB_RECORD_METHOD(void, SBBreakpointLocation, SetThreadName, (const char *),
                     thread_name);

  BreakpointLocationSP loc_sp = GetSP();
  if (loc_sp) {
    std::lock_guard<std::recursive_mutex> guard(
        loc_sp->GetTarget().GetAPIMutex());
    loc_sp->SetThreadName(thread_name);
  }
}

const char *SBBreakpointLocation::GetThreadName() const {
  LLDB_RECORD_METHOD_CONST_NO_ARGS(const char *, SBBreakpointLocation,
                                   GetThreadName);

  BreakpointLocationSP loc_sp = GetSP();
  if (loc_sp) {
    std::lock_guard<std::recursive_mutex> guard(
        loc_sp->GetTarget().GetAPIMutex());
    return loc_sp->GetThreadName();
  }
  return NULL;
}

void SBBreakpointLocation::SetQueueName(const char *queue_name) {
  LLDB_RECORD_METHOD(void, SBBreakpointLocation, SetQueueName, (const char *),
                     queue_name);

  BreakpointLocationSP loc_sp = GetSP();
  if (loc_sp) {
    std::lock_guard<std::recursive_mutex> guard(
        loc_sp->GetTarget().GetAPIMutex());
    loc_sp->SetQueueName(queue_name);
  }
}

const char *SBBreakpointLocation::GetQueueName() const {
  LLDB_RECORD_METHOD_CONST_NO_ARGS(const char *, SBBreakpointLocation,
                                   GetQueueName);

  BreakpointLocationSP loc_sp = GetSP();
  if (loc_sp) {
    std::lock_guard<std::recursive_mutex> guard(
        loc_sp->GetTarget().GetAPIMutex());
    loc_sp->GetQueueName();
  }
  return NULL;
}

bool SBBreakpointLocation::IsResolved() {
  LLDB_RECORD_METHOD_NO_ARGS(bool, SBBreakpointLocation, IsResolved);

  BreakpointLocationSP loc_sp = GetSP();
  if (loc_sp) {
    std::lock_guard<std::recursive_mutex> guard(
        loc_sp->GetTarget().GetAPIMutex());
    return loc_sp->IsResolved();
  }
  return false;
}

void SBBreakpointLocation::SetLocation(
    const lldb::BreakpointLocationSP &break_loc_sp) {
  // Uninstall the callbacks?
  m_opaque_wp = break_loc_sp;
}

bool SBBreakpointLocation::GetDescription(SBStream &description,
                                          DescriptionLevel level) {
  LLDB_RECORD_METHOD(bool, SBBreakpointLocation, GetDescription,
                     (lldb::SBStream &, lldb::DescriptionLevel), description,
                     level);

  Stream &strm = description.ref();
  BreakpointLocationSP loc_sp = GetSP();

  if (loc_sp) {
    std::lock_guard<std::recursive_mutex> guard(
        loc_sp->GetTarget().GetAPIMutex());
    loc_sp->GetDescription(&strm, level);
    strm.EOL();
  } else
    strm.PutCString("No value");

  return true;
}

break_id_t SBBreakpointLocation::GetID() {
  LLDB_RECORD_METHOD_NO_ARGS(lldb::break_id_t, SBBreakpointLocation, GetID);

  BreakpointLocationSP loc_sp = GetSP();
  if (loc_sp) {
    std::lock_guard<std::recursive_mutex> guard(
        loc_sp->GetTarget().GetAPIMutex());
    return loc_sp->GetID();
  } else
    return LLDB_INVALID_BREAK_ID;
}

SBBreakpoint SBBreakpointLocation::GetBreakpoint() {
  LLDB_RECORD_METHOD_NO_ARGS(lldb::SBBreakpoint, SBBreakpointLocation,
                             GetBreakpoint);

  Log *log(lldb_private::GetLogIfAllCategoriesSet(LIBLLDB_LOG_API));
  BreakpointLocationSP loc_sp = GetSP();

  SBBreakpoint sb_bp;
  if (loc_sp) {
    std::lock_guard<std::recursive_mutex> guard(
        loc_sp->GetTarget().GetAPIMutex());
    sb_bp = loc_sp->GetBreakpoint().shared_from_this();
  }

  if (log) {
    SBStream sstr;
    sb_bp.GetDescription(sstr);
    LLDB_LOG(log, "location = {0}, breakpoint = {1} ({2})", loc_sp.get(),
             sb_bp.GetSP().get(), sstr.GetData());
  }
  return LLDB_RECORD_RESULT(sb_bp);
}
