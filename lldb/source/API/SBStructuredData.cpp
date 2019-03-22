//===-- SBStructuredData.cpp ------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/API/SBStructuredData.h"
#include "SBReproducerPrivate.h"

#include "lldb/API/SBStream.h"
#include "lldb/API/SBStringList.h"
#include "lldb/Core/StructuredDataImpl.h"
#include "lldb/Target/StructuredDataPlugin.h"
#include "lldb/Utility/Event.h"
#include "lldb/Utility/Status.h"
#include "lldb/Utility/Stream.h"
#include "lldb/Utility/StructuredData.h"

using namespace lldb;
using namespace lldb_private;

#pragma mark--
#pragma mark SBStructuredData

SBStructuredData::SBStructuredData() : m_impl_up(new StructuredDataImpl()) {
  LLDB_RECORD_CONSTRUCTOR_NO_ARGS(SBStructuredData);
}

SBStructuredData::SBStructuredData(const lldb::SBStructuredData &rhs)
    : m_impl_up(new StructuredDataImpl(*rhs.m_impl_up.get())) {
  LLDB_RECORD_CONSTRUCTOR(SBStructuredData, (const lldb::SBStructuredData &),
                          rhs);
}

SBStructuredData::SBStructuredData(const lldb::EventSP &event_sp)
    : m_impl_up(new StructuredDataImpl(event_sp)) {
  LLDB_RECORD_CONSTRUCTOR(SBStructuredData, (const lldb::EventSP &), event_sp);
}

SBStructuredData::SBStructuredData(lldb_private::StructuredDataImpl *impl)
    : m_impl_up(impl) {
  LLDB_RECORD_CONSTRUCTOR(SBStructuredData,
                          (lldb_private::StructuredDataImpl *), impl);
}

SBStructuredData::~SBStructuredData() {}

SBStructuredData &SBStructuredData::
operator=(const lldb::SBStructuredData &rhs) {
  LLDB_RECORD_METHOD(
      lldb::SBStructuredData &,
      SBStructuredData, operator=,(const lldb::SBStructuredData &), rhs);

  *m_impl_up = *rhs.m_impl_up;
  return *this;
}

lldb::SBError SBStructuredData::SetFromJSON(lldb::SBStream &stream) {
  LLDB_RECORD_METHOD(lldb::SBError, SBStructuredData, SetFromJSON,
                     (lldb::SBStream &), stream);

  lldb::SBError error;
  std::string json_str(stream.GetData());

  StructuredData::ObjectSP json_obj = StructuredData::ParseJSON(json_str);
  m_impl_up->SetObjectSP(json_obj);

  if (!json_obj || json_obj->GetType() != eStructuredDataTypeDictionary)
    error.SetErrorString("Invalid Syntax");
  return LLDB_RECORD_RESULT(error);
}

bool SBStructuredData::IsValid() const {
  LLDB_RECORD_METHOD_CONST_NO_ARGS(bool, SBStructuredData, IsValid);

  return m_impl_up->IsValid();
}

void SBStructuredData::Clear() {
  LLDB_RECORD_METHOD_NO_ARGS(void, SBStructuredData, Clear);

  m_impl_up->Clear();
}

SBError SBStructuredData::GetAsJSON(lldb::SBStream &stream) const {
  LLDB_RECORD_METHOD_CONST(lldb::SBError, SBStructuredData, GetAsJSON,
                           (lldb::SBStream &), stream);

  SBError error;
  error.SetError(m_impl_up->GetAsJSON(stream.ref()));
  return LLDB_RECORD_RESULT(error);
}

lldb::SBError SBStructuredData::GetDescription(lldb::SBStream &stream) const {
  LLDB_RECORD_METHOD_CONST(lldb::SBError, SBStructuredData, GetDescription,
                           (lldb::SBStream &), stream);

  Status error = m_impl_up->GetDescription(stream.ref());
  SBError sb_error;
  sb_error.SetError(error);
  return LLDB_RECORD_RESULT(sb_error);
}

StructuredDataType SBStructuredData::GetType() const {
  LLDB_RECORD_METHOD_CONST_NO_ARGS(lldb::StructuredDataType, SBStructuredData,
                                   GetType);

  return (m_impl_up ? m_impl_up->GetType() : eStructuredDataTypeInvalid);
}

size_t SBStructuredData::GetSize() const {
  LLDB_RECORD_METHOD_CONST_NO_ARGS(size_t, SBStructuredData, GetSize);

  return (m_impl_up ? m_impl_up->GetSize() : 0);
}

bool SBStructuredData::GetKeys(lldb::SBStringList &keys) const {
  LLDB_RECORD_METHOD_CONST(bool, SBStructuredData, GetKeys,
                           (lldb::SBStringList &), keys);

  if (!m_impl_up)
    return false;
  
  if (GetType() != eStructuredDataTypeDictionary)
    return false;
  
  StructuredData::ObjectSP obj_sp = m_impl_up->GetObjectSP();
  if (!obj_sp)
    return false;

  StructuredData::Dictionary *dict = obj_sp->GetAsDictionary();
  // We claimed we were a dictionary, so this can't be null.
  assert(dict);
  // The return kind of GetKeys is an Array:
  StructuredData::ObjectSP array_sp = dict->GetKeys();
  StructuredData::Array *key_arr = array_sp->GetAsArray();
  assert(key_arr);
  
  key_arr->ForEach([&keys] (StructuredData::Object *object) -> bool {
    llvm::StringRef key = object->GetStringValue("");
    keys.AppendString(key.str().c_str());
    return true;
  });
  return true;
}

lldb::SBStructuredData SBStructuredData::GetValueForKey(const char *key) const {
  LLDB_RECORD_METHOD_CONST(lldb::SBStructuredData, SBStructuredData,
                           GetValueForKey, (const char *), key);

  if (!m_impl_up)
    return LLDB_RECORD_RESULT(SBStructuredData());

  SBStructuredData result;
  result.m_impl_up->SetObjectSP(m_impl_up->GetValueForKey(key));
  return LLDB_RECORD_RESULT(result);
}

lldb::SBStructuredData SBStructuredData::GetItemAtIndex(size_t idx) const {
  LLDB_RECORD_METHOD_CONST(lldb::SBStructuredData, SBStructuredData,
                           GetItemAtIndex, (size_t), idx);

  if (!m_impl_up)
    return LLDB_RECORD_RESULT(SBStructuredData());

  SBStructuredData result;
  result.m_impl_up->SetObjectSP(m_impl_up->GetItemAtIndex(idx));
  return LLDB_RECORD_RESULT(result);
}

uint64_t SBStructuredData::GetIntegerValue(uint64_t fail_value) const {
  LLDB_RECORD_METHOD_CONST(uint64_t, SBStructuredData, GetIntegerValue,
                           (uint64_t), fail_value);

  return (m_impl_up ? m_impl_up->GetIntegerValue(fail_value) : fail_value);
}

double SBStructuredData::GetFloatValue(double fail_value) const {
  LLDB_RECORD_METHOD_CONST(double, SBStructuredData, GetFloatValue, (double),
                           fail_value);

  return (m_impl_up ? m_impl_up->GetFloatValue(fail_value) : fail_value);
}

bool SBStructuredData::GetBooleanValue(bool fail_value) const {
  LLDB_RECORD_METHOD_CONST(bool, SBStructuredData, GetBooleanValue, (bool),
                           fail_value);

  return (m_impl_up ? m_impl_up->GetBooleanValue(fail_value) : fail_value);
}

size_t SBStructuredData::GetStringValue(char *dst, size_t dst_len) const {
  LLDB_RECORD_METHOD_CONST(size_t, SBStructuredData, GetStringValue,
                           (char *, size_t), dst, dst_len);

  return (m_impl_up ? m_impl_up->GetStringValue(dst, dst_len) : 0);
}
