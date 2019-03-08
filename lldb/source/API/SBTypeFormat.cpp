//===-- SBTypeFormat.cpp ------------------------------------------*- C++
//-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/API/SBTypeFormat.h"
#include "SBReproducerPrivate.h"

#include "lldb/API/SBStream.h"

#include "lldb/DataFormatters/DataVisualization.h"

using namespace lldb;
using namespace lldb_private;

SBTypeFormat::SBTypeFormat() : m_opaque_sp() {
  LLDB_RECORD_CONSTRUCTOR_NO_ARGS(SBTypeFormat);
}

SBTypeFormat::SBTypeFormat(lldb::Format format, uint32_t options)
    : m_opaque_sp(
          TypeFormatImplSP(new TypeFormatImpl_Format(format, options))) {
  LLDB_RECORD_CONSTRUCTOR(SBTypeFormat, (lldb::Format, uint32_t), format,
                          options);
}

SBTypeFormat::SBTypeFormat(const char *type, uint32_t options)
    : m_opaque_sp(TypeFormatImplSP(new TypeFormatImpl_EnumType(
          ConstString(type ? type : ""), options))) {
  LLDB_RECORD_CONSTRUCTOR(SBTypeFormat, (const char *, uint32_t), type,
                          options);
}

SBTypeFormat::SBTypeFormat(const lldb::SBTypeFormat &rhs)
    : m_opaque_sp(rhs.m_opaque_sp) {
  LLDB_RECORD_CONSTRUCTOR(SBTypeFormat, (const lldb::SBTypeFormat &), rhs);
}

SBTypeFormat::~SBTypeFormat() {}

bool SBTypeFormat::IsValid() const {
  LLDB_RECORD_METHOD_CONST_NO_ARGS(bool, SBTypeFormat, IsValid);

  return m_opaque_sp.get() != NULL;
}

lldb::Format SBTypeFormat::GetFormat() {
  LLDB_RECORD_METHOD_NO_ARGS(lldb::Format, SBTypeFormat, GetFormat);

  if (IsValid() && m_opaque_sp->GetType() == TypeFormatImpl::Type::eTypeFormat)
    return ((TypeFormatImpl_Format *)m_opaque_sp.get())->GetFormat();
  return lldb::eFormatInvalid;
}

const char *SBTypeFormat::GetTypeName() {
  LLDB_RECORD_METHOD_NO_ARGS(const char *, SBTypeFormat, GetTypeName);

  if (IsValid() && m_opaque_sp->GetType() == TypeFormatImpl::Type::eTypeEnum)
    return ((TypeFormatImpl_EnumType *)m_opaque_sp.get())
        ->GetTypeName()
        .AsCString("");
  return "";
}

uint32_t SBTypeFormat::GetOptions() {
  LLDB_RECORD_METHOD_NO_ARGS(uint32_t, SBTypeFormat, GetOptions);

  if (IsValid())
    return m_opaque_sp->GetOptions();
  return 0;
}

void SBTypeFormat::SetFormat(lldb::Format fmt) {
  LLDB_RECORD_METHOD(void, SBTypeFormat, SetFormat, (lldb::Format), fmt);

  if (CopyOnWrite_Impl(Type::eTypeFormat))
    ((TypeFormatImpl_Format *)m_opaque_sp.get())->SetFormat(fmt);
}

void SBTypeFormat::SetTypeName(const char *type) {
  LLDB_RECORD_METHOD(void, SBTypeFormat, SetTypeName, (const char *), type);

  if (CopyOnWrite_Impl(Type::eTypeEnum))
    ((TypeFormatImpl_EnumType *)m_opaque_sp.get())
        ->SetTypeName(ConstString(type ? type : ""));
}

void SBTypeFormat::SetOptions(uint32_t value) {
  LLDB_RECORD_METHOD(void, SBTypeFormat, SetOptions, (uint32_t), value);

  if (CopyOnWrite_Impl(Type::eTypeKeepSame))
    m_opaque_sp->SetOptions(value);
}

bool SBTypeFormat::GetDescription(lldb::SBStream &description,
                                  lldb::DescriptionLevel description_level) {
  LLDB_RECORD_METHOD(bool, SBTypeFormat, GetDescription,
                     (lldb::SBStream &, lldb::DescriptionLevel), description,
                     description_level);

  if (!IsValid())
    return false;
  else {
    description.Printf("%s\n", m_opaque_sp->GetDescription().c_str());
    return true;
  }
}

lldb::SBTypeFormat &SBTypeFormat::operator=(const lldb::SBTypeFormat &rhs) {
  LLDB_RECORD_METHOD(lldb::SBTypeFormat &,
                     SBTypeFormat, operator=,(const lldb::SBTypeFormat &), rhs);

  if (this != &rhs) {
    m_opaque_sp = rhs.m_opaque_sp;
  }
  return *this;
}

bool SBTypeFormat::operator==(lldb::SBTypeFormat &rhs) {
  LLDB_RECORD_METHOD(bool, SBTypeFormat, operator==,(lldb::SBTypeFormat &),
                     rhs);

  if (!IsValid())
    return !rhs.IsValid();
  return m_opaque_sp == rhs.m_opaque_sp;
}

bool SBTypeFormat::IsEqualTo(lldb::SBTypeFormat &rhs) {
  LLDB_RECORD_METHOD(bool, SBTypeFormat, IsEqualTo, (lldb::SBTypeFormat &),
                     rhs);

  if (!IsValid())
    return !rhs.IsValid();

  if (GetFormat() == rhs.GetFormat())
    return GetOptions() == rhs.GetOptions();
  else
    return false;
}

bool SBTypeFormat::operator!=(lldb::SBTypeFormat &rhs) {
  LLDB_RECORD_METHOD(bool, SBTypeFormat, operator!=,(lldb::SBTypeFormat &),
                     rhs);

  if (!IsValid())
    return !rhs.IsValid();
  return m_opaque_sp != rhs.m_opaque_sp;
}

lldb::TypeFormatImplSP SBTypeFormat::GetSP() { return m_opaque_sp; }

void SBTypeFormat::SetSP(const lldb::TypeFormatImplSP &typeformat_impl_sp) {
  m_opaque_sp = typeformat_impl_sp;
}

SBTypeFormat::SBTypeFormat(const lldb::TypeFormatImplSP &typeformat_impl_sp)
    : m_opaque_sp(typeformat_impl_sp) {}

bool SBTypeFormat::CopyOnWrite_Impl(Type type) {
  if (!IsValid())
    return false;

  if (m_opaque_sp.unique() &&
      ((type == Type::eTypeKeepSame) ||
       (type == Type::eTypeFormat &&
        m_opaque_sp->GetType() == TypeFormatImpl::Type::eTypeFormat) ||
       (type == Type::eTypeEnum &&
        m_opaque_sp->GetType() == TypeFormatImpl::Type::eTypeEnum)))
    return true;

  if (type == Type::eTypeKeepSame) {
    if (m_opaque_sp->GetType() == TypeFormatImpl::Type::eTypeFormat)
      type = Type::eTypeFormat;
    else
      type = Type::eTypeEnum;
  }

  if (type == Type::eTypeFormat)
    SetSP(
        TypeFormatImplSP(new TypeFormatImpl_Format(GetFormat(), GetOptions())));
  else
    SetSP(TypeFormatImplSP(
        new TypeFormatImpl_EnumType(ConstString(GetTypeName()), GetOptions())));

  return true;
}
