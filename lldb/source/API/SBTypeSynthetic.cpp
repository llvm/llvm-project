//===-- SBTypeSynthetic.cpp -----------------------------------------*- C++
//-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/API/SBTypeSynthetic.h"
#include "SBReproducerPrivate.h"

#include "lldb/API/SBStream.h"

#include "lldb/DataFormatters/DataVisualization.h"

using namespace lldb;
using namespace lldb_private;

#ifndef LLDB_DISABLE_PYTHON

SBTypeSynthetic::SBTypeSynthetic() : m_opaque_sp() {
  LLDB_RECORD_CONSTRUCTOR_NO_ARGS(SBTypeSynthetic);
}

SBTypeSynthetic SBTypeSynthetic::CreateWithClassName(const char *data,
                                                     uint32_t options) {
  LLDB_RECORD_STATIC_METHOD(lldb::SBTypeSynthetic, SBTypeSynthetic,
                            CreateWithClassName, (const char *, uint32_t), data,
                            options);

  if (!data || data[0] == 0)
    return LLDB_RECORD_RESULT(SBTypeSynthetic());
  return LLDB_RECORD_RESULT(SBTypeSynthetic(ScriptedSyntheticChildrenSP(
      new ScriptedSyntheticChildren(options, data, ""))));
}

SBTypeSynthetic SBTypeSynthetic::CreateWithScriptCode(const char *data,
                                                      uint32_t options) {
  LLDB_RECORD_STATIC_METHOD(lldb::SBTypeSynthetic, SBTypeSynthetic,
                            CreateWithScriptCode, (const char *, uint32_t),
                            data, options);

  if (!data || data[0] == 0)
    return LLDB_RECORD_RESULT(SBTypeSynthetic());
  return LLDB_RECORD_RESULT(SBTypeSynthetic(ScriptedSyntheticChildrenSP(
      new ScriptedSyntheticChildren(options, "", data))));
}

SBTypeSynthetic::SBTypeSynthetic(const lldb::SBTypeSynthetic &rhs)
    : m_opaque_sp(rhs.m_opaque_sp) {
  LLDB_RECORD_CONSTRUCTOR(SBTypeSynthetic, (const lldb::SBTypeSynthetic &),
                          rhs);
}

SBTypeSynthetic::~SBTypeSynthetic() {}

bool SBTypeSynthetic::IsValid() const {
  LLDB_RECORD_METHOD_CONST_NO_ARGS(bool, SBTypeSynthetic, IsValid);
  return this->operator bool();
}
SBTypeSynthetic::operator bool() const {
  LLDB_RECORD_METHOD_CONST_NO_ARGS(bool, SBTypeSynthetic, operator bool);

  return m_opaque_sp.get() != NULL;
}

bool SBTypeSynthetic::IsClassCode() {
  LLDB_RECORD_METHOD_NO_ARGS(bool, SBTypeSynthetic, IsClassCode);

  if (!IsValid())
    return false;
  const char *code = m_opaque_sp->GetPythonCode();
  return (code && *code);
}

bool SBTypeSynthetic::IsClassName() {
  LLDB_RECORD_METHOD_NO_ARGS(bool, SBTypeSynthetic, IsClassName);

  if (!IsValid())
    return false;
  return !IsClassCode();
}

const char *SBTypeSynthetic::GetData() {
  LLDB_RECORD_METHOD_NO_ARGS(const char *, SBTypeSynthetic, GetData);

  if (!IsValid())
    return NULL;
  if (IsClassCode())
    return m_opaque_sp->GetPythonCode();
  else
    return m_opaque_sp->GetPythonClassName();
}

void SBTypeSynthetic::SetClassName(const char *data) {
  LLDB_RECORD_METHOD(void, SBTypeSynthetic, SetClassName, (const char *), data);

  if (IsValid() && data && *data)
    m_opaque_sp->SetPythonClassName(data);
}

void SBTypeSynthetic::SetClassCode(const char *data) {
  LLDB_RECORD_METHOD(void, SBTypeSynthetic, SetClassCode, (const char *), data);

  if (IsValid() && data && *data)
    m_opaque_sp->SetPythonCode(data);
}

uint32_t SBTypeSynthetic::GetOptions() {
  LLDB_RECORD_METHOD_NO_ARGS(uint32_t, SBTypeSynthetic, GetOptions);

  if (!IsValid())
    return lldb::eTypeOptionNone;
  return m_opaque_sp->GetOptions();
}

void SBTypeSynthetic::SetOptions(uint32_t value) {
  LLDB_RECORD_METHOD(void, SBTypeSynthetic, SetOptions, (uint32_t), value);

  if (!CopyOnWrite_Impl())
    return;
  m_opaque_sp->SetOptions(value);
}

bool SBTypeSynthetic::GetDescription(lldb::SBStream &description,
                                     lldb::DescriptionLevel description_level) {
  LLDB_RECORD_METHOD(bool, SBTypeSynthetic, GetDescription,
                     (lldb::SBStream &, lldb::DescriptionLevel), description,
                     description_level);

  if (m_opaque_sp) {
    description.Printf("%s\n", m_opaque_sp->GetDescription().c_str());
    return true;
  }
  return false;
}

lldb::SBTypeSynthetic &SBTypeSynthetic::
operator=(const lldb::SBTypeSynthetic &rhs) {
  LLDB_RECORD_METHOD(lldb::SBTypeSynthetic &,
                     SBTypeSynthetic, operator=,(const lldb::SBTypeSynthetic &),
                     rhs);

  if (this != &rhs) {
    m_opaque_sp = rhs.m_opaque_sp;
  }
  return *this;
}

bool SBTypeSynthetic::operator==(lldb::SBTypeSynthetic &rhs) {
  LLDB_RECORD_METHOD(
      bool, SBTypeSynthetic, operator==,(lldb::SBTypeSynthetic &), rhs);

  if (!IsValid())
    return !rhs.IsValid();
  return m_opaque_sp == rhs.m_opaque_sp;
}

bool SBTypeSynthetic::IsEqualTo(lldb::SBTypeSynthetic &rhs) {
  LLDB_RECORD_METHOD(bool, SBTypeSynthetic, IsEqualTo,
                     (lldb::SBTypeSynthetic &), rhs);

  if (!IsValid())
    return !rhs.IsValid();

  if (m_opaque_sp->IsScripted() != rhs.m_opaque_sp->IsScripted())
    return false;

  if (IsClassCode() != rhs.IsClassCode())
    return false;

  if (strcmp(GetData(), rhs.GetData()))
    return false;

  return GetOptions() == rhs.GetOptions();
}

bool SBTypeSynthetic::operator!=(lldb::SBTypeSynthetic &rhs) {
  LLDB_RECORD_METHOD(
      bool, SBTypeSynthetic, operator!=,(lldb::SBTypeSynthetic &), rhs);

  if (!IsValid())
    return !rhs.IsValid();
  return m_opaque_sp != rhs.m_opaque_sp;
}

lldb::ScriptedSyntheticChildrenSP SBTypeSynthetic::GetSP() {
  return m_opaque_sp;
}

void SBTypeSynthetic::SetSP(
    const lldb::ScriptedSyntheticChildrenSP &TypeSynthetic_impl_sp) {
  m_opaque_sp = TypeSynthetic_impl_sp;
}

SBTypeSynthetic::SBTypeSynthetic(
    const lldb::ScriptedSyntheticChildrenSP &TypeSynthetic_impl_sp)
    : m_opaque_sp(TypeSynthetic_impl_sp) {}

bool SBTypeSynthetic::CopyOnWrite_Impl() {
  if (!IsValid())
    return false;
  if (m_opaque_sp.unique())
    return true;

  ScriptedSyntheticChildrenSP new_sp(new ScriptedSyntheticChildren(
      m_opaque_sp->GetOptions(), m_opaque_sp->GetPythonClassName(),
      m_opaque_sp->GetPythonCode()));

  SetSP(new_sp);

  return true;
}

#endif // LLDB_DISABLE_PYTHON
