//===-- SBSymbolContextList.cpp ---------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/API/SBSymbolContextList.h"
#include "SBReproducerPrivate.h"
#include "Utils.h"
#include "lldb/API/SBStream.h"
#include "lldb/Symbol/SymbolContext.h"

using namespace lldb;
using namespace lldb_private;

SBSymbolContextList::SBSymbolContextList()
    : m_opaque_up(new SymbolContextList()) {
  LLDB_RECORD_CONSTRUCTOR_NO_ARGS(SBSymbolContextList);
}

SBSymbolContextList::SBSymbolContextList(const SBSymbolContextList &rhs)
    : m_opaque_up() {
  LLDB_RECORD_CONSTRUCTOR(SBSymbolContextList,
                          (const lldb::SBSymbolContextList &), rhs);

  m_opaque_up = clone(rhs.m_opaque_up);
}

SBSymbolContextList::~SBSymbolContextList() {}

const SBSymbolContextList &SBSymbolContextList::
operator=(const SBSymbolContextList &rhs) {
  LLDB_RECORD_METHOD(
      const lldb::SBSymbolContextList &,
      SBSymbolContextList, operator=,(const lldb::SBSymbolContextList &), rhs);

  if (this != &rhs)
    m_opaque_up = clone(rhs.m_opaque_up);
  return *this;
}

uint32_t SBSymbolContextList::GetSize() const {
  LLDB_RECORD_METHOD_CONST_NO_ARGS(uint32_t, SBSymbolContextList, GetSize);

  if (m_opaque_up)
    return m_opaque_up->GetSize();
  return 0;
}

SBSymbolContext SBSymbolContextList::GetContextAtIndex(uint32_t idx) {
  LLDB_RECORD_METHOD(lldb::SBSymbolContext, SBSymbolContextList,
                     GetContextAtIndex, (uint32_t), idx);

  SBSymbolContext sb_sc;
  if (m_opaque_up) {
    SymbolContext sc;
    if (m_opaque_up->GetContextAtIndex(idx, sc)) {
      sb_sc.SetSymbolContext(&sc);
    }
  }
  return LLDB_RECORD_RESULT(sb_sc);
}

void SBSymbolContextList::Clear() {
  LLDB_RECORD_METHOD_NO_ARGS(void, SBSymbolContextList, Clear);

  if (m_opaque_up)
    m_opaque_up->Clear();
}

void SBSymbolContextList::Append(SBSymbolContext &sc) {
  LLDB_RECORD_METHOD(void, SBSymbolContextList, Append,
                     (lldb::SBSymbolContext &), sc);

  if (sc.IsValid() && m_opaque_up.get())
    m_opaque_up->Append(*sc);
}

void SBSymbolContextList::Append(SBSymbolContextList &sc_list) {
  LLDB_RECORD_METHOD(void, SBSymbolContextList, Append,
                     (lldb::SBSymbolContextList &), sc_list);

  if (sc_list.IsValid() && m_opaque_up.get())
    m_opaque_up->Append(*sc_list);
}

bool SBSymbolContextList::IsValid() const {
  LLDB_RECORD_METHOD_CONST_NO_ARGS(bool, SBSymbolContextList, IsValid);
  return this->operator bool();
}
SBSymbolContextList::operator bool() const {
  LLDB_RECORD_METHOD_CONST_NO_ARGS(bool, SBSymbolContextList, operator bool);

  return m_opaque_up != NULL;
}

lldb_private::SymbolContextList *SBSymbolContextList::operator->() const {
  return m_opaque_up.get();
}

lldb_private::SymbolContextList &SBSymbolContextList::operator*() const {
  assert(m_opaque_up.get());
  return *m_opaque_up;
}

bool SBSymbolContextList::GetDescription(lldb::SBStream &description) {
  LLDB_RECORD_METHOD(bool, SBSymbolContextList, GetDescription,
                     (lldb::SBStream &), description);

  Stream &strm = description.ref();
  if (m_opaque_up)
    m_opaque_up->GetDescription(&strm, lldb::eDescriptionLevelFull, NULL);
  return true;
}
