//===-- SBWatchpoint.cpp --------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/API/SBWatchpoint.h"
#include "SBReproducerPrivate.h"
#include "lldb/API/SBAddress.h"
#include "lldb/API/SBDebugger.h"
#include "lldb/API/SBDefines.h"
#include "lldb/API/SBEvent.h"
#include "lldb/API/SBStream.h"

#include "lldb/Breakpoint/Watchpoint.h"
#include "lldb/Breakpoint/WatchpointList.h"
#include "lldb/Core/StreamFile.h"
#include "lldb/Target/Process.h"
#include "lldb/Target/Target.h"
#include "lldb/Utility/Stream.h"
#include "lldb/lldb-defines.h"
#include "lldb/lldb-types.h"

using namespace lldb;
using namespace lldb_private;

SBWatchpoint::SBWatchpoint() { LLDB_RECORD_CONSTRUCTOR_NO_ARGS(SBWatchpoint); }

SBWatchpoint::SBWatchpoint(const lldb::WatchpointSP &wp_sp)
    : m_opaque_wp(wp_sp) {
  LLDB_RECORD_CONSTRUCTOR(SBWatchpoint, (const lldb::WatchpointSP &), wp_sp);
}

SBWatchpoint::SBWatchpoint(const SBWatchpoint &rhs)
    : m_opaque_wp(rhs.m_opaque_wp) {
  LLDB_RECORD_CONSTRUCTOR(SBWatchpoint, (const lldb::SBWatchpoint &), rhs);
}

const SBWatchpoint &SBWatchpoint::operator=(const SBWatchpoint &rhs) {
  LLDB_RECORD_METHOD(const lldb::SBWatchpoint &,
                     SBWatchpoint, operator=,(const lldb::SBWatchpoint &), rhs);

  m_opaque_wp = rhs.m_opaque_wp;
  return *this;
}

SBWatchpoint::~SBWatchpoint() {}

watch_id_t SBWatchpoint::GetID() {
  LLDB_RECORD_METHOD_NO_ARGS(lldb::watch_id_t, SBWatchpoint, GetID);


  watch_id_t watch_id = LLDB_INVALID_WATCH_ID;
  lldb::WatchpointSP watchpoint_sp(GetSP());
  if (watchpoint_sp)
    watch_id = watchpoint_sp->GetID();

  return watch_id;
}

bool SBWatchpoint::IsValid() const {
  LLDB_RECORD_METHOD_CONST_NO_ARGS(bool, SBWatchpoint, IsValid);
  return this->operator bool();
}
SBWatchpoint::operator bool() const {
  LLDB_RECORD_METHOD_CONST_NO_ARGS(bool, SBWatchpoint, operator bool);

  return bool(m_opaque_wp.lock());
}

SBError SBWatchpoint::GetError() {
  LLDB_RECORD_METHOD_NO_ARGS(lldb::SBError, SBWatchpoint, GetError);

  SBError sb_error;
  lldb::WatchpointSP watchpoint_sp(GetSP());
  if (watchpoint_sp) {
    sb_error.SetError(watchpoint_sp->GetError());
  }
  return LLDB_RECORD_RESULT(sb_error);
}

int32_t SBWatchpoint::GetHardwareIndex() {
  LLDB_RECORD_METHOD_NO_ARGS(int32_t, SBWatchpoint, GetHardwareIndex);

  int32_t hw_index = -1;

  lldb::WatchpointSP watchpoint_sp(GetSP());
  if (watchpoint_sp) {
    std::lock_guard<std::recursive_mutex> guard(
        watchpoint_sp->GetTarget().GetAPIMutex());
    hw_index = watchpoint_sp->GetHardwareIndex();
  }

  return hw_index;
}

addr_t SBWatchpoint::GetWatchAddress() {
  LLDB_RECORD_METHOD_NO_ARGS(lldb::addr_t, SBWatchpoint, GetWatchAddress);

  addr_t ret_addr = LLDB_INVALID_ADDRESS;

  lldb::WatchpointSP watchpoint_sp(GetSP());
  if (watchpoint_sp) {
    std::lock_guard<std::recursive_mutex> guard(
        watchpoint_sp->GetTarget().GetAPIMutex());
    ret_addr = watchpoint_sp->GetLoadAddress();
  }

  return ret_addr;
}

size_t SBWatchpoint::GetWatchSize() {
  LLDB_RECORD_METHOD_NO_ARGS(size_t, SBWatchpoint, GetWatchSize);

  size_t watch_size = 0;

  lldb::WatchpointSP watchpoint_sp(GetSP());
  if (watchpoint_sp) {
    std::lock_guard<std::recursive_mutex> guard(
        watchpoint_sp->GetTarget().GetAPIMutex());
    watch_size = watchpoint_sp->GetByteSize();
  }

  return watch_size;
}

void SBWatchpoint::SetEnabled(bool enabled) {
  LLDB_RECORD_METHOD(void, SBWatchpoint, SetEnabled, (bool), enabled);

  lldb::WatchpointSP watchpoint_sp(GetSP());
  if (watchpoint_sp) {
    Target &target = watchpoint_sp->GetTarget();
    std::lock_guard<std::recursive_mutex> guard(target.GetAPIMutex());
    ProcessSP process_sp = target.GetProcessSP();
    const bool notify = true;
    if (process_sp) {
      if (enabled)
        process_sp->EnableWatchpoint(watchpoint_sp.get(), notify);
      else
        process_sp->DisableWatchpoint(watchpoint_sp.get(), notify);
    } else {
      watchpoint_sp->SetEnabled(enabled, notify);
    }
  }
}

bool SBWatchpoint::IsEnabled() {
  LLDB_RECORD_METHOD_NO_ARGS(bool, SBWatchpoint, IsEnabled);

  lldb::WatchpointSP watchpoint_sp(GetSP());
  if (watchpoint_sp) {
    std::lock_guard<std::recursive_mutex> guard(
        watchpoint_sp->GetTarget().GetAPIMutex());
    return watchpoint_sp->IsEnabled();
  } else
    return false;
}

uint32_t SBWatchpoint::GetHitCount() {
  LLDB_RECORD_METHOD_NO_ARGS(uint32_t, SBWatchpoint, GetHitCount);

  uint32_t count = 0;
  lldb::WatchpointSP watchpoint_sp(GetSP());
  if (watchpoint_sp) {
    std::lock_guard<std::recursive_mutex> guard(
        watchpoint_sp->GetTarget().GetAPIMutex());
    count = watchpoint_sp->GetHitCount();
  }

  return count;
}

uint32_t SBWatchpoint::GetIgnoreCount() {
  LLDB_RECORD_METHOD_NO_ARGS(uint32_t, SBWatchpoint, GetIgnoreCount);

  lldb::WatchpointSP watchpoint_sp(GetSP());
  if (watchpoint_sp) {
    std::lock_guard<std::recursive_mutex> guard(
        watchpoint_sp->GetTarget().GetAPIMutex());
    return watchpoint_sp->GetIgnoreCount();
  } else
    return 0;
}

void SBWatchpoint::SetIgnoreCount(uint32_t n) {
  LLDB_RECORD_METHOD(void, SBWatchpoint, SetIgnoreCount, (uint32_t), n);

  lldb::WatchpointSP watchpoint_sp(GetSP());
  if (watchpoint_sp) {
    std::lock_guard<std::recursive_mutex> guard(
        watchpoint_sp->GetTarget().GetAPIMutex());
    watchpoint_sp->SetIgnoreCount(n);
  }
}

const char *SBWatchpoint::GetCondition() {
  LLDB_RECORD_METHOD_NO_ARGS(const char *, SBWatchpoint, GetCondition);

  lldb::WatchpointSP watchpoint_sp(GetSP());
  if (watchpoint_sp) {
    std::lock_guard<std::recursive_mutex> guard(
        watchpoint_sp->GetTarget().GetAPIMutex());
    return watchpoint_sp->GetConditionText();
  }
  return NULL;
}

void SBWatchpoint::SetCondition(const char *condition) {
  LLDB_RECORD_METHOD(void, SBWatchpoint, SetCondition, (const char *),
                     condition);

  lldb::WatchpointSP watchpoint_sp(GetSP());
  if (watchpoint_sp) {
    std::lock_guard<std::recursive_mutex> guard(
        watchpoint_sp->GetTarget().GetAPIMutex());
    watchpoint_sp->SetCondition(condition);
  }
}

bool SBWatchpoint::GetDescription(SBStream &description,
                                  DescriptionLevel level) {
  LLDB_RECORD_METHOD(bool, SBWatchpoint, GetDescription,
                     (lldb::SBStream &, lldb::DescriptionLevel), description,
                     level);

  Stream &strm = description.ref();

  lldb::WatchpointSP watchpoint_sp(GetSP());
  if (watchpoint_sp) {
    std::lock_guard<std::recursive_mutex> guard(
        watchpoint_sp->GetTarget().GetAPIMutex());
    watchpoint_sp->GetDescription(&strm, level);
    strm.EOL();
  } else
    strm.PutCString("No value");

  return true;
}

void SBWatchpoint::Clear() {
  LLDB_RECORD_METHOD_NO_ARGS(void, SBWatchpoint, Clear);

  m_opaque_wp.reset();
}

lldb::WatchpointSP SBWatchpoint::GetSP() const {
  LLDB_RECORD_METHOD_CONST_NO_ARGS(lldb::WatchpointSP, SBWatchpoint, GetSP);

  return LLDB_RECORD_RESULT(m_opaque_wp.lock());
}

void SBWatchpoint::SetSP(const lldb::WatchpointSP &sp) {
  LLDB_RECORD_METHOD(void, SBWatchpoint, SetSP, (const lldb::WatchpointSP &),
                     sp);

  m_opaque_wp = sp;
}

bool SBWatchpoint::EventIsWatchpointEvent(const lldb::SBEvent &event) {
  LLDB_RECORD_STATIC_METHOD(bool, SBWatchpoint, EventIsWatchpointEvent,
                            (const lldb::SBEvent &), event);

  return Watchpoint::WatchpointEventData::GetEventDataFromEvent(event.get()) !=
         NULL;
}

WatchpointEventType
SBWatchpoint::GetWatchpointEventTypeFromEvent(const SBEvent &event) {
  LLDB_RECORD_STATIC_METHOD(lldb::WatchpointEventType, SBWatchpoint,
                            GetWatchpointEventTypeFromEvent,
                            (const lldb::SBEvent &), event);

  if (event.IsValid())
    return Watchpoint::WatchpointEventData::GetWatchpointEventTypeFromEvent(
        event.GetSP());
  return eWatchpointEventTypeInvalidType;
}

SBWatchpoint SBWatchpoint::GetWatchpointFromEvent(const lldb::SBEvent &event) {
  LLDB_RECORD_STATIC_METHOD(lldb::SBWatchpoint, SBWatchpoint,
                            GetWatchpointFromEvent, (const lldb::SBEvent &),
                            event);

  SBWatchpoint sb_watchpoint;
  if (event.IsValid())
    sb_watchpoint =
        Watchpoint::WatchpointEventData::GetWatchpointFromEvent(event.GetSP());
  return LLDB_RECORD_RESULT(sb_watchpoint);
}
