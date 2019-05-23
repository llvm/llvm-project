//===-- SBQueueItem.cpp -----------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/lldb-forward.h"

#include "SBReproducerPrivate.h"
#include "lldb/API/SBAddress.h"
#include "lldb/API/SBQueueItem.h"
#include "lldb/API/SBThread.h"
#include "lldb/Core/Address.h"
#include "lldb/Target/Process.h"
#include "lldb/Target/QueueItem.h"
#include "lldb/Target/Thread.h"

using namespace lldb;
using namespace lldb_private;

// Constructors
SBQueueItem::SBQueueItem() : m_queue_item_sp() {
  LLDB_RECORD_CONSTRUCTOR_NO_ARGS(SBQueueItem);
}

SBQueueItem::SBQueueItem(const QueueItemSP &queue_item_sp)
    : m_queue_item_sp(queue_item_sp) {
  LLDB_RECORD_CONSTRUCTOR(SBQueueItem, (const lldb::QueueItemSP &),
                          queue_item_sp);
}

// Destructor
SBQueueItem::~SBQueueItem() { m_queue_item_sp.reset(); }

bool SBQueueItem::IsValid() const {
  LLDB_RECORD_METHOD_CONST_NO_ARGS(bool, SBQueueItem, IsValid);
  return this->operator bool();
}
SBQueueItem::operator bool() const {
  LLDB_RECORD_METHOD_CONST_NO_ARGS(bool, SBQueueItem, operator bool);

  return m_queue_item_sp.get() != nullptr;
}

void SBQueueItem::Clear() {
  LLDB_RECORD_METHOD_NO_ARGS(void, SBQueueItem, Clear);

  m_queue_item_sp.reset();
}

void SBQueueItem::SetQueueItem(const QueueItemSP &queue_item_sp) {
  LLDB_RECORD_METHOD(void, SBQueueItem, SetQueueItem,
                     (const lldb::QueueItemSP &), queue_item_sp);

  m_queue_item_sp = queue_item_sp;
}

lldb::QueueItemKind SBQueueItem::GetKind() const {
  LLDB_RECORD_METHOD_CONST_NO_ARGS(lldb::QueueItemKind, SBQueueItem, GetKind);

  QueueItemKind result = eQueueItemKindUnknown;
  if (m_queue_item_sp) {
    result = m_queue_item_sp->GetKind();
  }
  return result;
}

void SBQueueItem::SetKind(lldb::QueueItemKind kind) {
  LLDB_RECORD_METHOD(void, SBQueueItem, SetKind, (lldb::QueueItemKind), kind);

  if (m_queue_item_sp) {
    m_queue_item_sp->SetKind(kind);
  }
}

SBAddress SBQueueItem::GetAddress() const {
  LLDB_RECORD_METHOD_CONST_NO_ARGS(lldb::SBAddress, SBQueueItem, GetAddress);

  SBAddress result;
  if (m_queue_item_sp) {
    result.SetAddress(&m_queue_item_sp->GetAddress());
  }
  return LLDB_RECORD_RESULT(result);
}

void SBQueueItem::SetAddress(SBAddress addr) {
  LLDB_RECORD_METHOD(void, SBQueueItem, SetAddress, (lldb::SBAddress), addr);

  if (m_queue_item_sp) {
    m_queue_item_sp->SetAddress(addr.ref());
  }
}

SBThread SBQueueItem::GetExtendedBacktraceThread(const char *type) {
  LLDB_RECORD_METHOD(lldb::SBThread, SBQueueItem, GetExtendedBacktraceThread,
                     (const char *), type);

  SBThread result;
  if (m_queue_item_sp) {
    ProcessSP process_sp = m_queue_item_sp->GetProcessSP();
    Process::StopLocker stop_locker;
    if (process_sp && stop_locker.TryLock(&process_sp->GetRunLock())) {
      ThreadSP thread_sp;
      ConstString type_const(type);
      thread_sp = m_queue_item_sp->GetExtendedBacktraceThread(type_const);
      if (thread_sp) {
        // Save this in the Process' ExtendedThreadList so a strong pointer
        // retains the object
        process_sp->GetExtendedThreadList().AddThread(thread_sp);
        result.SetThread(thread_sp);
      }
    }
  }
  return LLDB_RECORD_RESULT(result);
}

namespace lldb_private {
namespace repro {

template <>
void RegisterMethods<SBQueueItem>(Registry &R) {
  LLDB_REGISTER_CONSTRUCTOR(SBQueueItem, ());
  LLDB_REGISTER_CONSTRUCTOR(SBQueueItem, (const lldb::QueueItemSP &));
  LLDB_REGISTER_METHOD_CONST(bool, SBQueueItem, IsValid, ());
  LLDB_REGISTER_METHOD_CONST(bool, SBQueueItem, operator bool, ());
  LLDB_REGISTER_METHOD(void, SBQueueItem, Clear, ());
  LLDB_REGISTER_METHOD(void, SBQueueItem, SetQueueItem,
                       (const lldb::QueueItemSP &));
  LLDB_REGISTER_METHOD_CONST(lldb::QueueItemKind, SBQueueItem, GetKind, ());
  LLDB_REGISTER_METHOD(void, SBQueueItem, SetKind, (lldb::QueueItemKind));
  LLDB_REGISTER_METHOD_CONST(lldb::SBAddress, SBQueueItem, GetAddress, ());
  LLDB_REGISTER_METHOD(void, SBQueueItem, SetAddress, (lldb::SBAddress));
  LLDB_REGISTER_METHOD(lldb::SBThread, SBQueueItem,
                       GetExtendedBacktraceThread, (const char *));
}

}
}
