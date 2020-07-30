//===-- SBStringList.cpp --------------------------------------------------===//
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
    m_opaque_up = std::make_unique<StringList>(*lldb_strings_ptr);
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
  return LLDB_RECORD_RESULT(*this);
}

SBStringList::~SBStringList() = default;

const lldb_private::StringList *SBStringList::operator->() const {
  return m_opaque_up.get();
}

const lldb_private::StringList &SBStringList::operator*() const {
  return *m_opaque_up;
}

bool SBStringList::IsValid() const {
  LLDB_RECORD_METHOD_CONST_NO_ARGS(bool, SBStringList, IsValid);
  return this->operator bool();
}
SBStringList::operator bool() const {
  LLDB_RECORD_METHOD_CONST_NO_ARGS(bool, SBStringList, operator bool);

  return (m_opaque_up != nullptr);
}

void SBStringList::AppendString(const char *str) {
  LLDB_RECORD_METHOD(void, SBStringList, AppendString, (const char *), str);

  if (str != nullptr) {
    if (IsValid())
      m_opaque_up->AppendString(str);
    else
      m_opaque_up = std::make_unique<lldb_private::StringList>(str);
  }
}

void SBStringList::AppendList(const char **strv, int strc) {
  LLDB_RECORD_METHOD(void, SBStringList, AppendList, (const char **, int), strv,
                     strc);

  if ((strv != nullptr) && (strc > 0)) {
    if (IsValid())
      m_opaque_up->AppendList(strv, strc);
    else
      m_opaque_up = std::make_unique<lldb_private::StringList>(strv, strc);
  }
}

void SBStringList::AppendList(const SBStringList &strings) {
  LLDB_RECORD_METHOD(void, SBStringList, AppendList,
                     (const lldb::SBStringList &), strings);

  if (strings.IsValid()) {
    if (!IsValid())
      m_opaque_up = std::make_unique<lldb_private::StringList>();
    m_opaque_up->AppendList(*(strings.m_opaque_up));
  }
}

void SBStringList::AppendList(const StringList &strings) {
  if (!IsValid())
    m_opaque_up = std::make_unique<lldb_private::StringList>();
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
  return nullptr;
}

const char *SBStringList::GetStringAtIndex(size_t idx) const {
  LLDB_RECORD_METHOD_CONST(const char *, SBStringList, GetStringAtIndex,
                           (size_t), idx);

  if (IsValid()) {
    return m_opaque_up->GetStringAtIndex(idx);
  }
  return nullptr;
}

void SBStringList::Clear() {
  LLDB_RECORD_METHOD_NO_ARGS(void, SBStringList, Clear);

  if (IsValid()) {
    m_opaque_up->Clear();
  }
}

namespace lldb_private {
namespace repro {

template <>
void RegisterMethods<SBStringList>(Registry &R) {
  LLDB_REGISTER_CONSTRUCTOR(SBStringList, ());
  LLDB_REGISTER_CONSTRUCTOR(SBStringList, (const lldb::SBStringList &));
  LLDB_REGISTER_METHOD(const lldb::SBStringList &,
                       SBStringList, operator=,(const lldb::SBStringList &));
  LLDB_REGISTER_METHOD_CONST(bool, SBStringList, IsValid, ());
  LLDB_REGISTER_METHOD_CONST(bool, SBStringList, operator bool, ());
  LLDB_REGISTER_METHOD(void, SBStringList, AppendString, (const char *));
  LLDB_REGISTER_METHOD(void, SBStringList, AppendList, (const char **, int));
  LLDB_REGISTER_METHOD(void, SBStringList, AppendList,
                       (const lldb::SBStringList &));
  LLDB_REGISTER_METHOD_CONST(uint32_t, SBStringList, GetSize, ());
  LLDB_REGISTER_METHOD(const char *, SBStringList, GetStringAtIndex,
                       (size_t));
  LLDB_REGISTER_METHOD_CONST(const char *, SBStringList, GetStringAtIndex,
                             (size_t));
  LLDB_REGISTER_METHOD(void, SBStringList, Clear, ());
}

}
}
