//===-- SBTypeNameSpecifier.cpp -------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/API/SBTypeNameSpecifier.h"
#include "lldb/Utility/ReproducerInstrumentation.h"

#include "lldb/API/SBStream.h"
#include "lldb/API/SBType.h"

#include "lldb/DataFormatters/DataVisualization.h"

using namespace lldb;
using namespace lldb_private;

SBTypeNameSpecifier::SBTypeNameSpecifier() {
  LLDB_RECORD_CONSTRUCTOR_NO_ARGS(SBTypeNameSpecifier);
}

SBTypeNameSpecifier::SBTypeNameSpecifier(const char *name, bool is_regex)
    : m_opaque_sp(new TypeNameSpecifierImpl(name, is_regex)) {
  LLDB_RECORD_CONSTRUCTOR(SBTypeNameSpecifier, (const char *, bool), name,
                          is_regex);

  if (name == nullptr || (*name) == 0)
    m_opaque_sp.reset();
}

SBTypeNameSpecifier::SBTypeNameSpecifier(SBType type) {
  LLDB_RECORD_CONSTRUCTOR(SBTypeNameSpecifier, (lldb::SBType), type);

  if (type.IsValid())
    m_opaque_sp = TypeNameSpecifierImplSP(
        new TypeNameSpecifierImpl(type.m_opaque_sp->GetCompilerType(true)));
}

SBTypeNameSpecifier::SBTypeNameSpecifier(const lldb::SBTypeNameSpecifier &rhs)
    : m_opaque_sp(rhs.m_opaque_sp) {
  LLDB_RECORD_CONSTRUCTOR(SBTypeNameSpecifier,
                          (const lldb::SBTypeNameSpecifier &), rhs);
}

SBTypeNameSpecifier::~SBTypeNameSpecifier() = default;

bool SBTypeNameSpecifier::IsValid() const {
  LLDB_RECORD_METHOD_CONST_NO_ARGS(bool, SBTypeNameSpecifier, IsValid);
  return this->operator bool();
}
SBTypeNameSpecifier::operator bool() const {
  LLDB_RECORD_METHOD_CONST_NO_ARGS(bool, SBTypeNameSpecifier, operator bool);

  return m_opaque_sp.get() != nullptr;
}

const char *SBTypeNameSpecifier::GetName() {
  LLDB_RECORD_METHOD_NO_ARGS(const char *, SBTypeNameSpecifier, GetName);

  if (!IsValid())
    return nullptr;

  return m_opaque_sp->GetName();
}

SBType SBTypeNameSpecifier::GetType() {
  LLDB_RECORD_METHOD_NO_ARGS(lldb::SBType, SBTypeNameSpecifier, GetType);

  if (!IsValid())
    return SBType();
  lldb_private::CompilerType c_type = m_opaque_sp->GetCompilerType();
  if (c_type.IsValid())
    return SBType(c_type);
  return SBType();
}

bool SBTypeNameSpecifier::IsRegex() {
  LLDB_RECORD_METHOD_NO_ARGS(bool, SBTypeNameSpecifier, IsRegex);

  if (!IsValid())
    return false;

  return m_opaque_sp->IsRegex();
}

bool SBTypeNameSpecifier::GetDescription(
    lldb::SBStream &description, lldb::DescriptionLevel description_level) {
  LLDB_RECORD_METHOD(bool, SBTypeNameSpecifier, GetDescription,
                     (lldb::SBStream &, lldb::DescriptionLevel), description,
                     description_level);

  if (!IsValid())
    return false;
  description.Printf("SBTypeNameSpecifier(%s,%s)", GetName(),
                     IsRegex() ? "regex" : "plain");
  return true;
}

lldb::SBTypeNameSpecifier &SBTypeNameSpecifier::
operator=(const lldb::SBTypeNameSpecifier &rhs) {
  LLDB_RECORD_METHOD(
      lldb::SBTypeNameSpecifier &,
      SBTypeNameSpecifier, operator=,(const lldb::SBTypeNameSpecifier &), rhs);

  if (this != &rhs) {
    m_opaque_sp = rhs.m_opaque_sp;
  }
  return *this;
}

bool SBTypeNameSpecifier::operator==(lldb::SBTypeNameSpecifier &rhs) {
  LLDB_RECORD_METHOD(
      bool, SBTypeNameSpecifier, operator==,(lldb::SBTypeNameSpecifier &), rhs);

  if (!IsValid())
    return !rhs.IsValid();
  return m_opaque_sp == rhs.m_opaque_sp;
}

bool SBTypeNameSpecifier::IsEqualTo(lldb::SBTypeNameSpecifier &rhs) {
  LLDB_RECORD_METHOD(bool, SBTypeNameSpecifier, IsEqualTo,
                     (lldb::SBTypeNameSpecifier &), rhs);

  if (!IsValid())
    return !rhs.IsValid();

  if (IsRegex() != rhs.IsRegex())
    return false;
  if (GetName() == nullptr || rhs.GetName() == nullptr)
    return false;

  return (strcmp(GetName(), rhs.GetName()) == 0);
}

bool SBTypeNameSpecifier::operator!=(lldb::SBTypeNameSpecifier &rhs) {
  LLDB_RECORD_METHOD(
      bool, SBTypeNameSpecifier, operator!=,(lldb::SBTypeNameSpecifier &), rhs);

  if (!IsValid())
    return !rhs.IsValid();
  return m_opaque_sp != rhs.m_opaque_sp;
}

lldb::TypeNameSpecifierImplSP SBTypeNameSpecifier::GetSP() {
  return m_opaque_sp;
}

void SBTypeNameSpecifier::SetSP(
    const lldb::TypeNameSpecifierImplSP &type_namespec_sp) {
  m_opaque_sp = type_namespec_sp;
}

SBTypeNameSpecifier::SBTypeNameSpecifier(
    const lldb::TypeNameSpecifierImplSP &type_namespec_sp)
    : m_opaque_sp(type_namespec_sp) {}
