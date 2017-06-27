//===-- SBStructuredData.cpp ------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/API/SBStructuredData.h"

#include "lldb/API/SBStream.h"
#include "lldb/Core/Event.h"
#include "lldb/Core/StructuredDataImpl.h"
#include "lldb/Target/StructuredDataPlugin.h"
#include "lldb/Utility/Status.h"
#include "lldb/Utility/Stream.h"
#include "lldb/Utility/StructuredData.h"

using namespace lldb;
using namespace lldb_private;

#pragma mark--
#pragma mark SBStructuredData

SBStructuredData::SBStructuredData() : m_impl_up(new StructuredDataImpl()) {}

SBStructuredData::SBStructuredData(const lldb::SBStructuredData &rhs)
    : m_impl_up(new StructuredDataImpl(*rhs.m_impl_up.get())) {}

SBStructuredData::SBStructuredData(const lldb::EventSP &event_sp)
    : m_impl_up(new StructuredDataImpl(event_sp)) {}

SBStructuredData::~SBStructuredData() {}

SBStructuredData &SBStructuredData::
operator=(const lldb::SBStructuredData &rhs) {
  *m_impl_up = *rhs.m_impl_up;
  return *this;
}

lldb::SBError SBStructuredData::SetFromJSON(lldb::SBStream &stream) {
  lldb::SBError error;
  std::string json_str(stream.GetData());

  StructuredData::ObjectSP json_obj = StructuredData::ParseJSON(json_str);
  m_impl_up->SetObjectSP(json_obj);

  if (!json_obj || json_obj->GetType() != eStructuredDataTypeDictionary)
    error.SetErrorString("Invalid Syntax");
  return error;
}

bool SBStructuredData::IsValid() const { return m_impl_up->IsValid(); }

void SBStructuredData::Clear() { m_impl_up->Clear(); }

SBError SBStructuredData::GetAsJSON(lldb::SBStream &stream) const {
  SBError error;
  error.SetError(m_impl_up->GetAsJSON(stream.ref()));
  return error;
}

lldb::SBError SBStructuredData::GetDescription(lldb::SBStream &stream) const {
  Status error = m_impl_up->GetDescription(stream.ref());
  SBError sb_error;
  sb_error.SetError(error);
  return sb_error;
}

StructuredDataType SBStructuredData::GetType() const {
  return (m_impl_up ? m_impl_up->GetType() : eStructuredDataTypeInvalid);
}

size_t SBStructuredData::GetSize() const {
  return (m_impl_up ? m_impl_up->GetSize() : 0);
}

lldb::SBStructuredData SBStructuredData::GetValueForKey(const char *key) const {
  if (!m_impl_up)
    return SBStructuredData();

  SBStructuredData result;
  result.m_impl_up->SetObjectSP(m_impl_up->GetValueForKey(key));
  return result;
}

lldb::SBStructuredData SBStructuredData::GetItemAtIndex(size_t idx) const {
  if (!m_impl_up)
    return SBStructuredData();

  SBStructuredData result;
  result.m_impl_up->SetObjectSP(m_impl_up->GetItemAtIndex(idx));
  return result;
}

uint64_t SBStructuredData::GetIntegerValue(uint64_t fail_value) const {
  return (m_impl_up ? m_impl_up->GetIntegerValue(fail_value) : fail_value);
}

double SBStructuredData::GetFloatValue(double fail_value) const {
  return (m_impl_up ? m_impl_up->GetFloatValue(fail_value) : fail_value);
}

bool SBStructuredData::GetBooleanValue(bool fail_value) const {
  return (m_impl_up ? m_impl_up->GetBooleanValue(fail_value) : fail_value);
}

size_t SBStructuredData::GetStringValue(char *dst, size_t dst_len) const {
  return (m_impl_up ? m_impl_up->GetStringValue(dst, dst_len) : 0);
}
