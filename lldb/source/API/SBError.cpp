//===-- SBError.cpp ---------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/API/SBError.h"
#include "SBReproducerPrivate.h"
#include "Utils.h"
#include "lldb/API/SBStream.h"
#include "lldb/Utility/Status.h"

#include <stdarg.h>

using namespace lldb;
using namespace lldb_private;

SBError::SBError() : m_opaque_up() { LLDB_RECORD_CONSTRUCTOR_NO_ARGS(SBError); }

SBError::SBError(const SBError &rhs) : m_opaque_up() {
  LLDB_RECORD_CONSTRUCTOR(SBError, (const lldb::SBError &), rhs);

  m_opaque_up = clone(rhs.m_opaque_up);
}

SBError::~SBError() {}

const SBError &SBError::operator=(const SBError &rhs) {
  LLDB_RECORD_METHOD(const lldb::SBError &,
                     SBError, operator=,(const lldb::SBError &), rhs);

  if (this != &rhs)
    m_opaque_up = clone(rhs.m_opaque_up);
  return LLDB_RECORD_RESULT(*this);
}

const char *SBError::GetCString() const {
  LLDB_RECORD_METHOD_CONST_NO_ARGS(const char *, SBError, GetCString);

  if (m_opaque_up)
    return m_opaque_up->AsCString();
  return nullptr;
}

void SBError::Clear() {
  LLDB_RECORD_METHOD_NO_ARGS(void, SBError, Clear);

  if (m_opaque_up)
    m_opaque_up->Clear();
}

bool SBError::Fail() const {
  LLDB_RECORD_METHOD_CONST_NO_ARGS(bool, SBError, Fail);

  bool ret_value = false;
  if (m_opaque_up)
    ret_value = m_opaque_up->Fail();


  return ret_value;
}

bool SBError::Success() const {
  LLDB_RECORD_METHOD_CONST_NO_ARGS(bool, SBError, Success);

  bool ret_value = true;
  if (m_opaque_up)
    ret_value = m_opaque_up->Success();

  return ret_value;
}

uint32_t SBError::GetError() const {
  LLDB_RECORD_METHOD_CONST_NO_ARGS(uint32_t, SBError, GetError);


  uint32_t err = 0;
  if (m_opaque_up)
    err = m_opaque_up->GetError();


  return err;
}

ErrorType SBError::GetType() const {
  LLDB_RECORD_METHOD_CONST_NO_ARGS(lldb::ErrorType, SBError, GetType);

  ErrorType err_type = eErrorTypeInvalid;
  if (m_opaque_up)
    err_type = m_opaque_up->GetType();

  return err_type;
}

void SBError::SetError(uint32_t err, ErrorType type) {
  LLDB_RECORD_METHOD(void, SBError, SetError, (uint32_t, lldb::ErrorType), err,
                     type);

  CreateIfNeeded();
  m_opaque_up->SetError(err, type);
}

void SBError::SetError(const Status &lldb_error) {
  CreateIfNeeded();
  *m_opaque_up = lldb_error;
}

void SBError::SetErrorToErrno() {
  LLDB_RECORD_METHOD_NO_ARGS(void, SBError, SetErrorToErrno);

  CreateIfNeeded();
  m_opaque_up->SetErrorToErrno();
}

void SBError::SetErrorToGenericError() {
  LLDB_RECORD_METHOD_NO_ARGS(void, SBError, SetErrorToGenericError);

  CreateIfNeeded();
  m_opaque_up->SetErrorToErrno();
}

void SBError::SetErrorString(const char *err_str) {
  LLDB_RECORD_METHOD(void, SBError, SetErrorString, (const char *), err_str);

  CreateIfNeeded();
  m_opaque_up->SetErrorString(err_str);
}

int SBError::SetErrorStringWithFormat(const char *format, ...) {
  CreateIfNeeded();
  va_list args;
  va_start(args, format);
  int num_chars = m_opaque_up->SetErrorStringWithVarArg(format, args);
  va_end(args);
  return num_chars;
}

bool SBError::IsValid() const {
  LLDB_RECORD_METHOD_CONST_NO_ARGS(bool, SBError, IsValid);
  return this->operator bool();
}
SBError::operator bool() const {
  LLDB_RECORD_METHOD_CONST_NO_ARGS(bool, SBError, operator bool);

  return m_opaque_up != nullptr;
}

void SBError::CreateIfNeeded() {
  if (m_opaque_up == nullptr)
    m_opaque_up.reset(new Status());
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
  LLDB_RECORD_METHOD(bool, SBError, GetDescription, (lldb::SBStream &),
                     description);

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

namespace lldb_private {
namespace repro {

template <>
void RegisterMethods<SBError>(Registry &R) {
  LLDB_REGISTER_CONSTRUCTOR(SBError, ());
  LLDB_REGISTER_CONSTRUCTOR(SBError, (const lldb::SBError &));
  LLDB_REGISTER_METHOD(const lldb::SBError &,
                       SBError, operator=,(const lldb::SBError &));
  LLDB_REGISTER_METHOD_CONST(const char *, SBError, GetCString, ());
  LLDB_REGISTER_METHOD(void, SBError, Clear, ());
  LLDB_REGISTER_METHOD_CONST(bool, SBError, Fail, ());
  LLDB_REGISTER_METHOD_CONST(bool, SBError, Success, ());
  LLDB_REGISTER_METHOD_CONST(uint32_t, SBError, GetError, ());
  LLDB_REGISTER_METHOD_CONST(lldb::ErrorType, SBError, GetType, ());
  LLDB_REGISTER_METHOD(void, SBError, SetError, (uint32_t, lldb::ErrorType));
  LLDB_REGISTER_METHOD(void, SBError, SetErrorToErrno, ());
  LLDB_REGISTER_METHOD(void, SBError, SetErrorToGenericError, ());
  LLDB_REGISTER_METHOD(void, SBError, SetErrorString, (const char *));
  LLDB_REGISTER_METHOD_CONST(bool, SBError, IsValid, ());
  LLDB_REGISTER_METHOD_CONST(bool, SBError, operator bool, ());
  LLDB_REGISTER_METHOD(bool, SBError, GetDescription, (lldb::SBStream &));
}

}
}
