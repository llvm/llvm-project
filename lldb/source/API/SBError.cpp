//===-- SBError.cpp -------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/API/SBError.h"
#include "Utils.h"
#include "lldb/API/SBStream.h"
#include "lldb/API/SBStructuredData.h"
#include "lldb/Core/StructuredDataImpl.h"
#include "lldb/Utility/Instrumentation.h"
#include "lldb/Utility/Status.h"
#include "lldb/Utility/VASPrintf.h"

#include <cstdarg>

using namespace lldb;
using namespace lldb_private;

SBError::SBError() { LLDB_INSTRUMENT_VA(this); }

SBError::SBError(const SBError &rhs) {
  LLDB_INSTRUMENT_VA(this, rhs);

  if (rhs.m_opaque_up)
    m_opaque_up = std::make_unique<Status>(rhs.m_opaque_up->Clone());
}

SBError::SBError(const char *message) {
  LLDB_INSTRUMENT_VA(this, message);

  SetErrorString(message);
}

SBError::SBError(lldb_private::Status &&status)
    : m_opaque_up(new Status(std::move(status))) {
  LLDB_INSTRUMENT_VA(this, status);
}

SBError::~SBError() = default;

const SBError &SBError::operator=(const SBError &rhs) {
  LLDB_INSTRUMENT_VA(this, rhs);

  if (this != &rhs)
    if (rhs.m_opaque_up)
      m_opaque_up = std::make_unique<Status>(rhs.m_opaque_up->Clone());

  return *this;
}

const char *SBError::GetCString() const {
  LLDB_INSTRUMENT_VA(this);

  if (m_opaque_up)
    return m_opaque_up->AsCString();
  return nullptr;
}

void SBError::Clear() {
  LLDB_INSTRUMENT_VA(this);

  if (m_opaque_up)
    m_opaque_up->Clear();
}

bool SBError::Fail() const {
  LLDB_INSTRUMENT_VA(this);

  bool ret_value = false;
  if (m_opaque_up)
    ret_value = m_opaque_up->Fail();


  return ret_value;
}

bool SBError::Success() const {
  LLDB_INSTRUMENT_VA(this);

  bool ret_value = true;
  if (m_opaque_up)
    ret_value = m_opaque_up->Success();

  return ret_value;
}

uint32_t SBError::GetError() const {
  LLDB_INSTRUMENT_VA(this);

  uint32_t err = 0;
  if (m_opaque_up)
    err = m_opaque_up->GetError();


  return err;
}

SBStructuredData SBError::GetErrorData() const {
  LLDB_INSTRUMENT_VA(this);

  SBStructuredData sb_data;
  if (!m_opaque_up)
    return sb_data;

  StructuredData::ObjectSP data(m_opaque_up->GetAsStructuredData());
  sb_data.m_impl_up->SetObjectSP(data);
  return sb_data;
}

ErrorType SBError::GetType() const {
  LLDB_INSTRUMENT_VA(this);

  ErrorType err_type = eErrorTypeInvalid;
  if (m_opaque_up)
    err_type = m_opaque_up->GetType();

  return err_type;
}

void SBError::SetError(uint32_t err, ErrorType type) {
  LLDB_INSTRUMENT_VA(this, err, type);

  CreateIfNeeded();
  *m_opaque_up = Status(err, type);
}

void SBError::SetError(Status &&lldb_error) {
  CreateIfNeeded();
  *m_opaque_up = std::move(lldb_error);
}

void SBError::SetErrorToErrno() {
  LLDB_INSTRUMENT_VA(this);

  CreateIfNeeded();
  *m_opaque_up = Status::FromErrno();
}

void SBError::SetErrorToGenericError() {
  LLDB_INSTRUMENT_VA(this);

  CreateIfNeeded();
  *m_opaque_up = Status(std::string("generic error"));
}

void SBError::SetErrorString(const char *err_str) {
  LLDB_INSTRUMENT_VA(this, err_str);

  CreateIfNeeded();
  *m_opaque_up = Status::FromErrorString(err_str);
}

int SBError::SetErrorStringWithFormat(const char *format, ...) {
  CreateIfNeeded();
  std::string string;
  va_list args;
  va_start(args, format);
  if (format != nullptr && format[0]) {
    llvm::SmallString<1024> buf;
    VASprintf(buf, format, args);
    string = std::string(buf.str());
    *m_opaque_up = Status(std::move(string));
  }
  va_end(args);
  return string.size();
}

bool SBError::IsValid() const {
  LLDB_INSTRUMENT_VA(this);
  return this->operator bool();
}
SBError::operator bool() const {
  LLDB_INSTRUMENT_VA(this);

  return m_opaque_up != nullptr;
}

void SBError::CreateIfNeeded() {
  if (m_opaque_up == nullptr)
    m_opaque_up = std::make_unique<Status>();
}

lldb_private::Status *SBError::operator->() { return m_opaque_up.get(); }

lldb_private::Status *SBError::get() { return m_opaque_up.get(); }

lldb_private::Status &SBError::ref() {
  CreateIfNeeded();
  return *m_opaque_up;
}

const lldb_private::Status &SBError::operator*() const {
  // Be sure to call "IsValid()" before calling this function or it will crash
  return *m_opaque_up;
}

bool SBError::GetDescription(SBStream &description) {
  LLDB_INSTRUMENT_VA(this, description);

  if (m_opaque_up) {
    if (m_opaque_up->Success())
      description.Printf("success");
    else {
      const char *err_string = GetCString();
      description.Printf("error: %s",
                         (err_string != nullptr ? err_string : ""));
    }
  } else
    description.Printf("error: <NULL>");

  return true;
}
