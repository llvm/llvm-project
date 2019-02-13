//===-- SBStringList.cpp ----------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/API/SBStringList.h"

#include "lldb/Utility/StringList.h"

using namespace lldb;
using namespace lldb_private;

SBStringList::SBStringList() : m_opaque_up() {}

SBStringList::SBStringList(const lldb_private::StringList *lldb_strings_ptr)
    : m_opaque_up() {
  if (lldb_strings_ptr)
    m_opaque_up.reset(new lldb_private::StringList(*lldb_strings_ptr));
}

SBStringList::SBStringList(const SBStringList &rhs) : m_opaque_up() {
  if (rhs.IsValid())
    m_opaque_up.reset(new lldb_private::StringList(*rhs));
}

const SBStringList &SBStringList::operator=(const SBStringList &rhs) {
  if (this != &rhs) {
    if (rhs.IsValid())
      m_opaque_up.reset(new lldb_private::StringList(*rhs));
    else
      m_opaque_up.reset();
  }
  return *this;
}

SBStringList::~SBStringList() {}

const lldb_private::StringList *SBStringList::operator->() const {
  return m_opaque_up.get();
}

const lldb_private::StringList &SBStringList::operator*() const {
  return *m_opaque_up;
}

bool SBStringList::IsValid() const { return (m_opaque_up != NULL); }

void SBStringList::AppendString(const char *str) {
  if (str != NULL) {
    if (IsValid())
      m_opaque_up->AppendString(str);
    else
      m_opaque_up.reset(new lldb_private::StringList(str));
  }
}

void SBStringList::AppendList(const char **strv, int strc) {
  if ((strv != NULL) && (strc > 0)) {
    if (IsValid())
      m_opaque_up->AppendList(strv, strc);
    else
      m_opaque_up.reset(new lldb_private::StringList(strv, strc));
  }
}

void SBStringList::AppendList(const SBStringList &strings) {
  if (strings.IsValid()) {
    if (!IsValid())
      m_opaque_up.reset(new lldb_private::StringList());
    m_opaque_up->AppendList(*(strings.m_opaque_up));
  }
}

void SBStringList::AppendList(const StringList &strings) {
  if (!IsValid())
    m_opaque_up.reset(new lldb_private::StringList());
  m_opaque_up->AppendList(strings);
}

uint32_t SBStringList::GetSize() const {
  if (IsValid()) {
    return m_opaque_up->GetSize();
  }
  return 0;
}

const char *SBStringList::GetStringAtIndex(size_t idx) {
  if (IsValid()) {
    return m_opaque_up->GetStringAtIndex(idx);
  }
  return NULL;
}

const char *SBStringList::GetStringAtIndex(size_t idx) const {
  if (IsValid()) {
    return m_opaque_up->GetStringAtIndex(idx);
  }
  return NULL;
}

void SBStringList::Clear() {
  if (IsValid()) {
    m_opaque_up->Clear();
  }
}
