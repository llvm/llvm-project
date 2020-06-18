//===-- SBEnvironment.cpp -------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/API/SBEnvironment.h"
#include "SBReproducerPrivate.h"
#include "Utils.h"
#include "lldb/API/SBStringList.h"
#include "lldb/Utility/ConstString.h"
#include "lldb/Utility/Environment.h"

using namespace lldb;
using namespace lldb_private;

SBEnvironment::SBEnvironment() : m_opaque_up(new Environment()) {
  LLDB_RECORD_CONSTRUCTOR_NO_ARGS(SBEnvironment);
}

SBEnvironment::SBEnvironment(const SBEnvironment &rhs)
    : m_opaque_up(clone(rhs.m_opaque_up)) {
  LLDB_RECORD_CONSTRUCTOR(SBEnvironment, (const lldb::SBEnvironment &), rhs);
}

SBEnvironment::SBEnvironment(Environment rhs)
    : m_opaque_up(new Environment(std::move(rhs))) {}

SBEnvironment::~SBEnvironment() = default;

const SBEnvironment &SBEnvironment::operator=(const SBEnvironment &rhs) {
  LLDB_RECORD_METHOD(const lldb::SBEnvironment &,
                     SBEnvironment, operator=,(const lldb::SBEnvironment &),
                     rhs);

  if (this != &rhs)
    m_opaque_up = clone(rhs.m_opaque_up);
  return LLDB_RECORD_RESULT(*this);
}

size_t SBEnvironment::GetNumValues() {
  LLDB_RECORD_METHOD_NO_ARGS(size_t, SBEnvironment, GetNumValues);

  return m_opaque_up->size();
}

const char *SBEnvironment::Get(const char *name) {
  LLDB_RECORD_METHOD(const char *, SBEnvironment, Get, (const char *), name);

  auto entry = m_opaque_up->find(name);
  if (entry == m_opaque_up->end()) {
    return nullptr;
  }
  return ConstString(entry->second).AsCString("");
}

const char *SBEnvironment::GetNameAtIndex(size_t index) {
  LLDB_RECORD_METHOD(const char *, SBEnvironment, GetNameAtIndex, (size_t),
                     index);

  if (index >= GetNumValues())
    return nullptr;
  return ConstString(std::next(m_opaque_up->begin(), index)->first())
      .AsCString("");
}

const char *SBEnvironment::GetValueAtIndex(size_t index) {
  LLDB_RECORD_METHOD(const char *, SBEnvironment, GetValueAtIndex, (size_t),
                     index);

  if (index >= GetNumValues())
    return nullptr;
  return ConstString(std::next(m_opaque_up->begin(), index)->second)
      .AsCString("");
}

bool SBEnvironment::Set(const char *name, const char *value, bool overwrite) {
  LLDB_RECORD_METHOD(bool, SBEnvironment, Set,
                     (const char *, const char *, bool), name, value,
                     overwrite);

  if (overwrite) {
    m_opaque_up->insert_or_assign(name, std::string(value));
    return true;
  }
  return m_opaque_up->try_emplace(name, std::string(value)).second;
}

bool SBEnvironment::Unset(const char *name) {
  LLDB_RECORD_METHOD(bool, SBEnvironment, Unset, (const char *), name);

  return m_opaque_up->erase(name);
}

SBStringList SBEnvironment::GetEntries() {
  LLDB_RECORD_METHOD_NO_ARGS(lldb::SBStringList, SBEnvironment, GetEntries);

  SBStringList entries;
  for (const auto &KV : *m_opaque_up) {
    entries.AppendString(Environment::compose(KV).c_str());
  }
  return LLDB_RECORD_RESULT(entries);
}

void SBEnvironment::PutEntry(const char *name_and_value) {
  LLDB_RECORD_METHOD(void, SBEnvironment, PutEntry, (const char *),
                     name_and_value);

  auto split = llvm::StringRef(name_and_value).split('=');
  m_opaque_up->insert_or_assign(split.first.str(), split.second.str());
}

void SBEnvironment::SetEntries(const SBStringList &entries, bool append) {
  LLDB_RECORD_METHOD(void, SBEnvironment, SetEntries,
                     (const lldb::SBStringList &, bool), entries, append);

  if (!append)
    m_opaque_up->clear();
  for (size_t i = 0; i < entries.GetSize(); i++) {
    PutEntry(entries.GetStringAtIndex(i));
  }
}

void SBEnvironment::Clear() {
  LLDB_RECORD_METHOD_NO_ARGS(void, SBEnvironment, Clear);

  m_opaque_up->clear();
}

Environment &SBEnvironment::ref() const { return *m_opaque_up; }

namespace lldb_private {
namespace repro {
template <> void RegisterMethods<SBEnvironment>(Registry &R) {
  LLDB_REGISTER_CONSTRUCTOR(SBEnvironment, ());
  LLDB_REGISTER_CONSTRUCTOR(SBEnvironment, (const lldb::SBEnvironment &));
  LLDB_REGISTER_METHOD(const lldb::SBEnvironment &,
                       SBEnvironment, operator=,(const lldb::SBEnvironment &));
  LLDB_REGISTER_METHOD(size_t, SBEnvironment, GetNumValues, ());
  LLDB_REGISTER_METHOD(const char *, SBEnvironment, Get, (const char *));
  LLDB_REGISTER_METHOD(const char *, SBEnvironment, GetNameAtIndex, (size_t));
  LLDB_REGISTER_METHOD(const char *, SBEnvironment, GetValueAtIndex, (size_t));
  LLDB_REGISTER_METHOD(bool, SBEnvironment, Set,
                       (const char *, const char *, bool));
  LLDB_REGISTER_METHOD(bool, SBEnvironment, Unset, (const char *));
  LLDB_REGISTER_METHOD(lldb::SBStringList, SBEnvironment, GetEntries, ());
  LLDB_REGISTER_METHOD(void, SBEnvironment, PutEntry, (const char *));
  LLDB_REGISTER_METHOD(void, SBEnvironment, SetEntries,
                       (const lldb::SBStringList &, bool));
  LLDB_REGISTER_METHOD(void, SBEnvironment, Clear, ());
}
} // namespace repro
} // namespace lldb_private
