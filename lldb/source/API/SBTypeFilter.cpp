//===-- SBTypeFilter.cpp --------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/API/SBTypeFilter.h"
#include "SBReproducerPrivate.h"

#include "lldb/API/SBStream.h"

#include "lldb/DataFormatters/DataVisualization.h"

using namespace lldb;
using namespace lldb_private;

SBTypeFilter::SBTypeFilter() : m_opaque_sp() {
  LLDB_RECORD_CONSTRUCTOR_NO_ARGS(SBTypeFilter);
}

SBTypeFilter::SBTypeFilter(uint32_t options)
    : m_opaque_sp(TypeFilterImplSP(new TypeFilterImpl(options))) {
  LLDB_RECORD_CONSTRUCTOR(SBTypeFilter, (uint32_t), options);
}

SBTypeFilter::SBTypeFilter(const lldb::SBTypeFilter &rhs)
    : m_opaque_sp(rhs.m_opaque_sp) {
  LLDB_RECORD_CONSTRUCTOR(SBTypeFilter, (const lldb::SBTypeFilter &), rhs);
}

SBTypeFilter::~SBTypeFilter() = default;

bool SBTypeFilter::IsValid() const {
  LLDB_RECORD_METHOD_CONST_NO_ARGS(bool, SBTypeFilter, IsValid);
  return this->operator bool();
}
SBTypeFilter::operator bool() const {
  LLDB_RECORD_METHOD_CONST_NO_ARGS(bool, SBTypeFilter, operator bool);

  return m_opaque_sp.get() != nullptr;
}

uint32_t SBTypeFilter::GetOptions() {
  LLDB_RECORD_METHOD_NO_ARGS(uint32_t, SBTypeFilter, GetOptions);

  if (IsValid())
    return m_opaque_sp->GetOptions();
  return 0;
}

void SBTypeFilter::SetOptions(uint32_t value) {
  LLDB_RECORD_METHOD(void, SBTypeFilter, SetOptions, (uint32_t), value);

  if (CopyOnWrite_Impl())
    m_opaque_sp->SetOptions(value);
}

bool SBTypeFilter::GetDescription(lldb::SBStream &description,
                                  lldb::DescriptionLevel description_level) {
  LLDB_RECORD_METHOD(bool, SBTypeFilter, GetDescription,
                     (lldb::SBStream &, lldb::DescriptionLevel), description,
                     description_level);

  if (!IsValid())
    return false;
  else {
    description.Printf("%s\n", m_opaque_sp->GetDescription().c_str());
    return true;
  }
}

void SBTypeFilter::Clear() {
  LLDB_RECORD_METHOD_NO_ARGS(void, SBTypeFilter, Clear);

  if (CopyOnWrite_Impl())
    m_opaque_sp->Clear();
}

uint32_t SBTypeFilter::GetNumberOfExpressionPaths() {
  LLDB_RECORD_METHOD_NO_ARGS(uint32_t, SBTypeFilter,
                             GetNumberOfExpressionPaths);

  if (IsValid())
    return m_opaque_sp->GetCount();
  return 0;
}

const char *SBTypeFilter::GetExpressionPathAtIndex(uint32_t i) {
  LLDB_RECORD_METHOD(const char *, SBTypeFilter, GetExpressionPathAtIndex,
                     (uint32_t), i);

  if (IsValid()) {
    const char *item = m_opaque_sp->GetExpressionPathAtIndex(i);
    if (item && *item == '.')
      item++;
    return item;
  }
  return nullptr;
}

bool SBTypeFilter::ReplaceExpressionPathAtIndex(uint32_t i, const char *item) {
  LLDB_RECORD_METHOD(bool, SBTypeFilter, ReplaceExpressionPathAtIndex,
                     (uint32_t, const char *), i, item);

  if (CopyOnWrite_Impl())
    return m_opaque_sp->SetExpressionPathAtIndex(i, item);
  else
    return false;
}

void SBTypeFilter::AppendExpressionPath(const char *item) {
  LLDB_RECORD_METHOD(void, SBTypeFilter, AppendExpressionPath, (const char *),
                     item);

  if (CopyOnWrite_Impl())
    m_opaque_sp->AddExpressionPath(item);
}

lldb::SBTypeFilter &SBTypeFilter::operator=(const lldb::SBTypeFilter &rhs) {
  LLDB_RECORD_METHOD(lldb::SBTypeFilter &,
                     SBTypeFilter, operator=,(const lldb::SBTypeFilter &), rhs);

  if (this != &rhs) {
    m_opaque_sp = rhs.m_opaque_sp;
  }
  return LLDB_RECORD_RESULT(*this);
}

bool SBTypeFilter::operator==(lldb::SBTypeFilter &rhs) {
  LLDB_RECORD_METHOD(bool, SBTypeFilter, operator==,(lldb::SBTypeFilter &),
                     rhs);

  if (!IsValid())
    return !rhs.IsValid();

  return m_opaque_sp == rhs.m_opaque_sp;
}

bool SBTypeFilter::IsEqualTo(lldb::SBTypeFilter &rhs) {
  LLDB_RECORD_METHOD(bool, SBTypeFilter, IsEqualTo, (lldb::SBTypeFilter &),
                     rhs);

  if (!IsValid())
    return !rhs.IsValid();

  if (GetNumberOfExpressionPaths() != rhs.GetNumberOfExpressionPaths())
    return false;

  for (uint32_t j = 0; j < GetNumberOfExpressionPaths(); j++)
    if (strcmp(GetExpressionPathAtIndex(j), rhs.GetExpressionPathAtIndex(j)) !=
        0)
      return false;

  return GetOptions() == rhs.GetOptions();
}

bool SBTypeFilter::operator!=(lldb::SBTypeFilter &rhs) {
  LLDB_RECORD_METHOD(bool, SBTypeFilter, operator!=,(lldb::SBTypeFilter &),
                     rhs);

  if (!IsValid())
    return !rhs.IsValid();

  return m_opaque_sp != rhs.m_opaque_sp;
}

lldb::TypeFilterImplSP SBTypeFilter::GetSP() { return m_opaque_sp; }

void SBTypeFilter::SetSP(const lldb::TypeFilterImplSP &typefilter_impl_sp) {
  m_opaque_sp = typefilter_impl_sp;
}

SBTypeFilter::SBTypeFilter(const lldb::TypeFilterImplSP &typefilter_impl_sp)
    : m_opaque_sp(typefilter_impl_sp) {}

bool SBTypeFilter::CopyOnWrite_Impl() {
  if (!IsValid())
    return false;
  if (m_opaque_sp.unique())
    return true;

  TypeFilterImplSP new_sp(new TypeFilterImpl(GetOptions()));

  for (uint32_t j = 0; j < GetNumberOfExpressionPaths(); j++)
    new_sp->AddExpressionPath(GetExpressionPathAtIndex(j));

  SetSP(new_sp);

  return true;
}

namespace lldb_private {
namespace repro {

template <>
void RegisterMethods<SBTypeFilter>(Registry &R) {
  LLDB_REGISTER_CONSTRUCTOR(SBTypeFilter, ());
  LLDB_REGISTER_CONSTRUCTOR(SBTypeFilter, (uint32_t));
  LLDB_REGISTER_CONSTRUCTOR(SBTypeFilter, (const lldb::SBTypeFilter &));
  LLDB_REGISTER_METHOD_CONST(bool, SBTypeFilter, IsValid, ());
  LLDB_REGISTER_METHOD_CONST(bool, SBTypeFilter, operator bool, ());
  LLDB_REGISTER_METHOD(uint32_t, SBTypeFilter, GetOptions, ());
  LLDB_REGISTER_METHOD(void, SBTypeFilter, SetOptions, (uint32_t));
  LLDB_REGISTER_METHOD(bool, SBTypeFilter, GetDescription,
                       (lldb::SBStream &, lldb::DescriptionLevel));
  LLDB_REGISTER_METHOD(void, SBTypeFilter, Clear, ());
  LLDB_REGISTER_METHOD(uint32_t, SBTypeFilter, GetNumberOfExpressionPaths,
                       ());
  LLDB_REGISTER_METHOD(const char *, SBTypeFilter, GetExpressionPathAtIndex,
                       (uint32_t));
  LLDB_REGISTER_METHOD(bool, SBTypeFilter, ReplaceExpressionPathAtIndex,
                       (uint32_t, const char *));
  LLDB_REGISTER_METHOD(void, SBTypeFilter, AppendExpressionPath,
                       (const char *));
  LLDB_REGISTER_METHOD(lldb::SBTypeFilter &,
                       SBTypeFilter, operator=,(const lldb::SBTypeFilter &));
  LLDB_REGISTER_METHOD(bool, SBTypeFilter, operator==,(lldb::SBTypeFilter &));
  LLDB_REGISTER_METHOD(bool, SBTypeFilter, IsEqualTo, (lldb::SBTypeFilter &));
  LLDB_REGISTER_METHOD(bool, SBTypeFilter, operator!=,(lldb::SBTypeFilter &));
}

}
}
