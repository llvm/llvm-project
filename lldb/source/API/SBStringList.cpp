//===-- SBStringList.cpp ----------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/API/SBStringList.h"
#include "SBReproducerPrivate.h"
#include "Utils.h"
#include "lldb/Utility/StringList.h"

using namespace lldb;
using namespace lldb_private;

SBStringList::SBStringList() : m_opaque_up() {
  LLDB_RECORD_CONSTRUCTOR_NO_ARGS(SBStringList);
}

SBStringList::SBStringList(const lldb_private::StringList *lldb_strings_ptr)
    : m_opaque_up() {
  if (lldb_strings_ptr)
    m_opaque_up = llvm::make_unique<StringList>(*lldb_strings_ptr);
}

SBStringList::SBStringList(const SBStringList &rhs) : m_opaque_up() {
  LLDB_RECORD_CONSTRUCTOR(SBStringList, (const lldb::SBStringList &), rhs);

  m_opaque_up = clone(rhs.m_opaque_up);
}

const SBStringList &SBStringList::operator=(const SBStringList &rhs) {
  LLDB_RECORD_METHOD(const lldb::SBStringList &,
                     SBStringList, operator=,(const lldb::SBStringList &), rhs);

  if (this != &rhs)
    m_opaque_up = clone(rhs.m_opaque_up);
  return *this;
}

SBStringList::~SBStringList() {}

const lldb_private::StringList *SBStringList::operator->() const {
  return m_opaque_up.get();
}

const lldb_private::StringList &SBStringList::operator*() const {
  return *m_opaque_up;
}

bool SBStringList::IsValid() const {
  LLDB_RECORD_METHOD_CONST_NO_ARGS(bool, SBStringList, IsValid);

  return (m_opaque_up != NULL);
}

void SBStringList::AppendString(const char *str) {
  LLDB_RECORD_METHOD(void, SBStringList, AppendString, (const char *), str);

  if (str != NULL) {
    if (IsValid())
      m_opaque_up->AppendString(str);
    else
      m_opaque_up.reset(new lldb_private::StringList(str));
  }
}

void SBStringList::AppendList(const char **strv, int strc) {
  LLDB_RECORD_METHOD(void, SBStringList, AppendList, (const char **, int), strv,
                     strc);

  if ((strv != NULL) && (strc > 0)) {
    if (IsValid())
      m_opaque_up->AppendList(strv, strc);
    else
      m_opaque_up.reset(new lldb_private::StringList(strv, strc));
  }
}

void SBStringList::AppendList(const SBStringList &strings) {
  LLDB_RECORD_METHOD(void, SBStringList, AppendList,
                     (const lldb::SBStringList &), strings);

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
  LLDB_RECORD_METHOD_CONST_NO_ARGS(uint32_t, SBStringList, GetSize);

  if (IsValid()) {
    return m_opaque_up->GetSize();
  }
  return 0;
}

const char *SBStringList::GetStringAtIndex(size_t idx) {
  LLDB_RECORD_METHOD(const char *, SBStringList, GetStringAtIndex, (size_t),
                     idx);

  if (IsValid()) {
    return m_opaque_up->GetStringAtIndex(idx);
  }
  return NULL;
}

const char *SBStringList::GetStringAtIndex(size_t idx) const {
  LLDB_RECORD_METHOD_CONST(const char *, SBStringList, GetStringAtIndex,
                           (size_t), idx);

  if (IsValid()) {
    return m_opaque_up->GetStringAtIndex(idx);
  }
  return NULL;
}

void SBStringList::Clear() {
  LLDB_RECORD_METHOD_NO_ARGS(void, SBStringList, Clear);

  if (IsValid()) {
    m_opaque_up->Clear();
  }
}
