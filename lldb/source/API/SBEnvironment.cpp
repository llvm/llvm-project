//===-- SBEnvironment.cpp -------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/API/SBEnvironment.h"
#include "Utils.h"
#include "lldb/API/SBStringList.h"
#include "lldb/Utility/ConstString.h"
#include "lldb/Utility/Environment.h"

using namespace lldb;
using namespace lldb_private;

/// This class is highly mutable, therefore we don't reproducers.

SBEnvironment::SBEnvironment() : m_opaque_up(new Environment()) {}

SBEnvironment::SBEnvironment(const SBEnvironment &rhs)
    : m_opaque_up(clone(rhs.m_opaque_up)) {}

SBEnvironment::SBEnvironment(Environment rhs)
    : m_opaque_up(new Environment(std::move(rhs))) {}

SBEnvironment::~SBEnvironment() = default;

const SBEnvironment &SBEnvironment::operator=(const SBEnvironment &rhs) {
  if (this != &rhs)
    m_opaque_up = clone(rhs.m_opaque_up);
  return *this;
}

size_t SBEnvironment::GetNumValues() {
  return m_opaque_up->size();
}

const char *SBEnvironment::Get(const char *name) {
  auto entry = m_opaque_up->find(name);
  if (entry == m_opaque_up->end()) {
    return nullptr;
  }
  return ConstString(entry->second).AsCString("");
}

const char *SBEnvironment::GetNameAtIndex(size_t index) {
  if (index >= GetNumValues())
    return nullptr;
  return ConstString(std::next(m_opaque_up->begin(), index)->first())
      .AsCString("");
}

const char *SBEnvironment::GetValueAtIndex(size_t index) {
  if (index >= GetNumValues())
    return nullptr;
  return ConstString(std::next(m_opaque_up->begin(), index)->second)
      .AsCString("");
}

bool SBEnvironment::Set(const char *name, const char *value, bool overwrite) {
  if (overwrite) {
    m_opaque_up->insert_or_assign(name, std::string(value));
    return true;
  }
  return m_opaque_up->try_emplace(name, std::string(value)).second;
}

bool SBEnvironment::Unset(const char *name) {
  return m_opaque_up->erase(name);
}

SBStringList SBEnvironment::GetEntries() {
  SBStringList entries;
  for (const auto &KV : *m_opaque_up) {
    entries.AppendString(Environment::compose(KV).c_str());
  }
  return entries;
}

void SBEnvironment::PutEntry(const char *name_and_value) {
  auto split = llvm::StringRef(name_and_value).split('=');
  m_opaque_up->insert_or_assign(split.first.str(), split.second.str());
}

void SBEnvironment::SetEntries(const SBStringList &entries, bool append) {
  if (!append)
    m_opaque_up->clear();
  for (size_t i = 0; i < entries.GetSize(); i++) {
    PutEntry(entries.GetStringAtIndex(i));
  }
}

void SBEnvironment::Clear() {
  m_opaque_up->clear();
}

Environment &SBEnvironment::ref() const { return *m_opaque_up; }
