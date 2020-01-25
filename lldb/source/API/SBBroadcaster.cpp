//===-- SBBroadcaster.cpp -------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "SBReproducerPrivate.h"
#include "lldb/Utility/Broadcaster.h"

#include "lldb/API/SBBroadcaster.h"
#include "lldb/API/SBEvent.h"
#include "lldb/API/SBListener.h"

using namespace lldb;
using namespace lldb_private;

SBBroadcaster::SBBroadcaster() : m_opaque_sp(), m_opaque_ptr(nullptr) {
  LLDB_RECORD_CONSTRUCTOR_NO_ARGS(SBBroadcaster);
}

SBBroadcaster::SBBroadcaster(const char *name)
    : m_opaque_sp(new Broadcaster(nullptr, name)), m_opaque_ptr(nullptr) {
  LLDB_RECORD_CONSTRUCTOR(SBBroadcaster, (const char *), name);

  m_opaque_ptr = m_opaque_sp.get();
}

SBBroadcaster::SBBroadcaster(lldb_private::Broadcaster *broadcaster, bool owns)
    : m_opaque_sp(owns ? broadcaster : nullptr), m_opaque_ptr(broadcaster) {}

SBBroadcaster::SBBroadcaster(const SBBroadcaster &rhs)
    : m_opaque_sp(rhs.m_opaque_sp), m_opaque_ptr(rhs.m_opaque_ptr) {
  LLDB_RECORD_CONSTRUCTOR(SBBroadcaster, (const lldb::SBBroadcaster &), rhs);
}

const SBBroadcaster &SBBroadcaster::operator=(const SBBroadcaster &rhs) {
  LLDB_RECORD_METHOD(const lldb::SBBroadcaster &,
                     SBBroadcaster, operator=,(const lldb::SBBroadcaster &),
                     rhs);

  if (this != &rhs) {
    m_opaque_sp = rhs.m_opaque_sp;
    m_opaque_ptr = rhs.m_opaque_ptr;
  }
  return LLDB_RECORD_RESULT(*this);
}

SBBroadcaster::~SBBroadcaster() { reset(nullptr, false); }

void SBBroadcaster::BroadcastEventByType(uint32_t event_type, bool unique) {
  LLDB_RECORD_METHOD(void, SBBroadcaster, BroadcastEventByType,
                     (uint32_t, bool), event_type, unique);

  if (m_opaque_ptr == nullptr)
    return;

  if (unique)
    m_opaque_ptr->BroadcastEventIfUnique(event_type);
  else
    m_opaque_ptr->BroadcastEvent(event_type);
}

void SBBroadcaster::BroadcastEvent(const SBEvent &event, bool unique) {
  LLDB_RECORD_METHOD(void, SBBroadcaster, BroadcastEvent,
                     (const lldb::SBEvent &, bool), event, unique);

  if (m_opaque_ptr == nullptr)
    return;

  EventSP event_sp = event.GetSP();
  if (unique)
    m_opaque_ptr->BroadcastEventIfUnique(event_sp);
  else
    m_opaque_ptr->BroadcastEvent(event_sp);
}

void SBBroadcaster::AddInitialEventsToListener(const SBListener &listener,
                                               uint32_t requested_events) {
  LLDB_RECORD_METHOD(void, SBBroadcaster, AddInitialEventsToListener,
                     (const lldb::SBListener &, uint32_t), listener,
                     requested_events);

  if (m_opaque_ptr)
    m_opaque_ptr->AddInitialEventsToListener(listener.m_opaque_sp,
                                             requested_events);
}

uint32_t SBBroadcaster::AddListener(const SBListener &listener,
                                    uint32_t event_mask) {
  LLDB_RECORD_METHOD(uint32_t, SBBroadcaster, AddListener,
                     (const lldb::SBListener &, uint32_t), listener,
                     event_mask);

  if (m_opaque_ptr)
    return m_opaque_ptr->AddListener(listener.m_opaque_sp, event_mask);
  return 0;
}

const char *SBBroadcaster::GetName() const {
  LLDB_RECORD_METHOD_CONST_NO_ARGS(const char *, SBBroadcaster, GetName);

  if (m_opaque_ptr)
    return m_opaque_ptr->GetBroadcasterName().GetCString();
  return nullptr;
}

bool SBBroadcaster::EventTypeHasListeners(uint32_t event_type) {
  LLDB_RECORD_METHOD(bool, SBBroadcaster, EventTypeHasListeners, (uint32_t),
                     event_type);

  if (m_opaque_ptr)
    return m_opaque_ptr->EventTypeHasListeners(event_type);
  return false;
}

bool SBBroadcaster::RemoveListener(const SBListener &listener,
                                   uint32_t event_mask) {
  LLDB_RECORD_METHOD(bool, SBBroadcaster, RemoveListener,
                     (const lldb::SBListener &, uint32_t), listener,
                     event_mask);

  if (m_opaque_ptr)
    return m_opaque_ptr->RemoveListener(listener.m_opaque_sp, event_mask);
  return false;
}

Broadcaster *SBBroadcaster::get() const { return m_opaque_ptr; }

void SBBroadcaster::reset(Broadcaster *broadcaster, bool owns) {
  if (owns)
    m_opaque_sp.reset(broadcaster);
  else
    m_opaque_sp.reset();
  m_opaque_ptr = broadcaster;
}

bool SBBroadcaster::IsValid() const {
  LLDB_RECORD_METHOD_CONST_NO_ARGS(bool, SBBroadcaster, IsValid);
  return this->operator bool();
}
SBBroadcaster::operator bool() const {
  LLDB_RECORD_METHOD_CONST_NO_ARGS(bool, SBBroadcaster, operator bool);

  return m_opaque_ptr != nullptr;
}

void SBBroadcaster::Clear() {
  LLDB_RECORD_METHOD_NO_ARGS(void, SBBroadcaster, Clear);

  m_opaque_sp.reset();
  m_opaque_ptr = nullptr;
}

bool SBBroadcaster::operator==(const SBBroadcaster &rhs) const {
  LLDB_RECORD_METHOD_CONST(
      bool, SBBroadcaster, operator==,(const lldb::SBBroadcaster &), rhs);

  return m_opaque_ptr == rhs.m_opaque_ptr;
}

bool SBBroadcaster::operator!=(const SBBroadcaster &rhs) const {
  LLDB_RECORD_METHOD_CONST(
      bool, SBBroadcaster, operator!=,(const lldb::SBBroadcaster &), rhs);

  return m_opaque_ptr != rhs.m_opaque_ptr;
}

bool SBBroadcaster::operator<(const SBBroadcaster &rhs) const {
  LLDB_RECORD_METHOD_CONST(
      bool, SBBroadcaster, operator<,(const lldb::SBBroadcaster &), rhs);

  return m_opaque_ptr < rhs.m_opaque_ptr;
}

namespace lldb_private {
namespace repro {

template <>
void RegisterMethods<SBBroadcaster>(Registry &R) {
  LLDB_REGISTER_CONSTRUCTOR(SBBroadcaster, ());
  LLDB_REGISTER_CONSTRUCTOR(SBBroadcaster, (const char *));
  LLDB_REGISTER_CONSTRUCTOR(SBBroadcaster, (const lldb::SBBroadcaster &));
  LLDB_REGISTER_METHOD(
      const lldb::SBBroadcaster &,
      SBBroadcaster, operator=,(const lldb::SBBroadcaster &));
  LLDB_REGISTER_METHOD(void, SBBroadcaster, BroadcastEventByType,
                       (uint32_t, bool));
  LLDB_REGISTER_METHOD(void, SBBroadcaster, BroadcastEvent,
                       (const lldb::SBEvent &, bool));
  LLDB_REGISTER_METHOD(void, SBBroadcaster, AddInitialEventsToListener,
                       (const lldb::SBListener &, uint32_t));
  LLDB_REGISTER_METHOD(uint32_t, SBBroadcaster, AddListener,
                       (const lldb::SBListener &, uint32_t));
  LLDB_REGISTER_METHOD_CONST(const char *, SBBroadcaster, GetName, ());
  LLDB_REGISTER_METHOD(bool, SBBroadcaster, EventTypeHasListeners,
                       (uint32_t));
  LLDB_REGISTER_METHOD(bool, SBBroadcaster, RemoveListener,
                       (const lldb::SBListener &, uint32_t));
  LLDB_REGISTER_METHOD_CONST(bool, SBBroadcaster, IsValid, ());
  LLDB_REGISTER_METHOD_CONST(bool, SBBroadcaster, operator bool, ());
  LLDB_REGISTER_METHOD(void, SBBroadcaster, Clear, ());
  LLDB_REGISTER_METHOD_CONST(
      bool, SBBroadcaster, operator==,(const lldb::SBBroadcaster &));
  LLDB_REGISTER_METHOD_CONST(
      bool, SBBroadcaster, operator!=,(const lldb::SBBroadcaster &));
  LLDB_REGISTER_METHOD_CONST(
      bool, SBBroadcaster, operator<,(const lldb::SBBroadcaster &));
}

}
}
