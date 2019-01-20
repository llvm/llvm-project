//===-- SBSymbolContextList.cpp ---------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/API/SBSymbolContextList.h"
#include "lldb/API/SBStream.h"
#include "lldb/Symbol/SymbolContext.h"

using namespace lldb;
using namespace lldb_private;

SBSymbolContextList::SBSymbolContextList()
    : m_opaque_ap(new SymbolContextList()) {}

SBSymbolContextList::SBSymbolContextList(const SBSymbolContextList &rhs)
    : m_opaque_ap(new SymbolContextList(*rhs.m_opaque_ap)) {}

SBSymbolContextList::~SBSymbolContextList() {}

const SBSymbolContextList &SBSymbolContextList::
operator=(const SBSymbolContextList &rhs) {
  if (this != &rhs) {
    *m_opaque_ap = *rhs.m_opaque_ap;
  }
  return *this;
}

uint32_t SBSymbolContextList::GetSize() const {
  if (m_opaque_ap)
    return m_opaque_ap->GetSize();
  return 0;
}

SBSymbolContext SBSymbolContextList::GetContextAtIndex(uint32_t idx) {
  SBSymbolContext sb_sc;
  if (m_opaque_ap) {
    SymbolContext sc;
    if (m_opaque_ap->GetContextAtIndex(idx, sc)) {
      sb_sc.SetSymbolContext(&sc);
    }
  }
  return sb_sc;
}

void SBSymbolContextList::Clear() {
  if (m_opaque_ap)
    m_opaque_ap->Clear();
}

void SBSymbolContextList::Append(SBSymbolContext &sc) {
  if (sc.IsValid() && m_opaque_ap.get())
    m_opaque_ap->Append(*sc);
}

void SBSymbolContextList::Append(SBSymbolContextList &sc_list) {
  if (sc_list.IsValid() && m_opaque_ap.get())
    m_opaque_ap->Append(*sc_list);
}

bool SBSymbolContextList::IsValid() const { return m_opaque_ap != NULL; }

lldb_private::SymbolContextList *SBSymbolContextList::operator->() const {
  return m_opaque_ap.get();
}

lldb_private::SymbolContextList &SBSymbolContextList::operator*() const {
  assert(m_opaque_ap.get());
  return *m_opaque_ap;
}

bool SBSymbolContextList::GetDescription(lldb::SBStream &description) {
  Stream &strm = description.ref();
  if (m_opaque_ap)
    m_opaque_ap->GetDescription(&strm, lldb::eDescriptionLevelFull, NULL);
  return true;
}
