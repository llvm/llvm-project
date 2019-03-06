//===-- SBTraceOptions.cpp --------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/API/SBTraceOptions.h"
#include "SBReproducerPrivate.h"
#include "lldb/API/SBError.h"
#include "lldb/API/SBStructuredData.h"
#include "lldb/Core/StructuredDataImpl.h"
#include "lldb/Utility/Log.h"
#include "lldb/Utility/TraceOptions.h"

#include <memory>

using namespace lldb;
using namespace lldb_private;

SBTraceOptions::SBTraceOptions() {
  LLDB_RECORD_CONSTRUCTOR_NO_ARGS(SBTraceOptions);

  m_traceoptions_sp = std::make_shared<TraceOptions>();
}

lldb::TraceType SBTraceOptions::getType() const {
  LLDB_RECORD_METHOD_CONST_NO_ARGS(lldb::TraceType, SBTraceOptions, getType);

  if (m_traceoptions_sp)
    return m_traceoptions_sp->getType();
  return lldb::TraceType::eTraceTypeNone;
}

uint64_t SBTraceOptions::getTraceBufferSize() const {
  LLDB_RECORD_METHOD_CONST_NO_ARGS(uint64_t, SBTraceOptions,
                                   getTraceBufferSize);

  if (m_traceoptions_sp)
    return m_traceoptions_sp->getTraceBufferSize();
  return 0;
}

lldb::SBStructuredData SBTraceOptions::getTraceParams(lldb::SBError &error) {
  LLDB_RECORD_METHOD(lldb::SBStructuredData, SBTraceOptions, getTraceParams,
                     (lldb::SBError &), error);

  error.Clear();
  const lldb_private::StructuredData::DictionarySP dict_obj =
      m_traceoptions_sp->getTraceParams();
  lldb::SBStructuredData structData;
  if (dict_obj && structData.m_impl_up)
    structData.m_impl_up->SetObjectSP(dict_obj->shared_from_this());
  else
    error.SetErrorString("Empty trace params");
  return LLDB_RECORD_RESULT(structData);
}

uint64_t SBTraceOptions::getMetaDataBufferSize() const {
  LLDB_RECORD_METHOD_CONST_NO_ARGS(uint64_t, SBTraceOptions,
                                   getMetaDataBufferSize);

  if (m_traceoptions_sp)
    return m_traceoptions_sp->getTraceBufferSize();
  return 0;
}

void SBTraceOptions::setTraceParams(lldb::SBStructuredData &params) {
  LLDB_RECORD_METHOD(void, SBTraceOptions, setTraceParams,
                     (lldb::SBStructuredData &), params);

  if (m_traceoptions_sp && params.m_impl_up) {
    StructuredData::ObjectSP obj_sp = params.m_impl_up->GetObjectSP();
    if (obj_sp && obj_sp->GetAsDictionary() != nullptr)
      m_traceoptions_sp->setTraceParams(
          std::static_pointer_cast<StructuredData::Dictionary>(obj_sp));
  }
  return;
}

void SBTraceOptions::setType(lldb::TraceType type) {
  LLDB_RECORD_METHOD(void, SBTraceOptions, setType, (lldb::TraceType), type);

  if (m_traceoptions_sp)
    m_traceoptions_sp->setType(type);
}

void SBTraceOptions::setTraceBufferSize(uint64_t size) {
  LLDB_RECORD_METHOD(void, SBTraceOptions, setTraceBufferSize, (uint64_t),
                     size);

  if (m_traceoptions_sp)
    m_traceoptions_sp->setTraceBufferSize(size);
}

void SBTraceOptions::setMetaDataBufferSize(uint64_t size) {
  LLDB_RECORD_METHOD(void, SBTraceOptions, setMetaDataBufferSize, (uint64_t),
                     size);

  if (m_traceoptions_sp)
    m_traceoptions_sp->setMetaDataBufferSize(size);
}

bool SBTraceOptions::IsValid() {
  LLDB_RECORD_METHOD_NO_ARGS(bool, SBTraceOptions, IsValid);

  if (m_traceoptions_sp)
    return true;
  return false;
}

void SBTraceOptions::setThreadID(lldb::tid_t thread_id) {
  LLDB_RECORD_METHOD(void, SBTraceOptions, setThreadID, (lldb::tid_t),
                     thread_id);

  if (m_traceoptions_sp)
    m_traceoptions_sp->setThreadID(thread_id);
}

lldb::tid_t SBTraceOptions::getThreadID() {
  LLDB_RECORD_METHOD_NO_ARGS(lldb::tid_t, SBTraceOptions, getThreadID);

  if (m_traceoptions_sp)
    return m_traceoptions_sp->getThreadID();
  return LLDB_INVALID_THREAD_ID;
}
