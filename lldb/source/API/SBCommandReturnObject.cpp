//===-- SBCommandReturnObject.cpp -------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/API/SBCommandReturnObject.h"
#include "SBReproducerPrivate.h"
#include "Utils.h"
#include "lldb/API/SBError.h"
#include "lldb/API/SBFile.h"
#include "lldb/API/SBStream.h"
#include "lldb/Interpreter/CommandReturnObject.h"
#include "lldb/Utility/ConstString.h"
#include "lldb/Utility/Status.h"

using namespace lldb;
using namespace lldb_private;

class lldb_private::SBCommandReturnObjectImpl {
public:
  SBCommandReturnObjectImpl()
      : m_ptr(new CommandReturnObject()), m_owned(true) {}
  SBCommandReturnObjectImpl(CommandReturnObject &ref)
      : m_ptr(&ref), m_owned(false) {}
  SBCommandReturnObjectImpl(const SBCommandReturnObjectImpl &rhs)
      : m_ptr(new CommandReturnObject(*rhs.m_ptr)), m_owned(rhs.m_owned) {}
  SBCommandReturnObjectImpl &operator=(const SBCommandReturnObjectImpl &rhs) {
    SBCommandReturnObjectImpl copy(rhs);
    std::swap(*this, copy);
    return *this;
  }
  // rvalue ctor+assignment are not used by SBCommandReturnObject.
  ~SBCommandReturnObjectImpl() {
    if (m_owned)
      delete m_ptr;
  }

  CommandReturnObject &operator*() const { return *m_ptr; }

private:
  CommandReturnObject *m_ptr;
  bool m_owned;
};

SBCommandReturnObject::SBCommandReturnObject()
    : m_opaque_up(new SBCommandReturnObjectImpl()) {
  LLDB_RECORD_CONSTRUCTOR_NO_ARGS(SBCommandReturnObject);
}

SBCommandReturnObject::SBCommandReturnObject(CommandReturnObject &ref)
    : m_opaque_up(new SBCommandReturnObjectImpl(ref)) {
  LLDB_RECORD_CONSTRUCTOR(SBCommandReturnObject,
                          (lldb_private::CommandReturnObject &), ref);
}

SBCommandReturnObject::SBCommandReturnObject(const SBCommandReturnObject &rhs)
    : m_opaque_up() {
  LLDB_RECORD_CONSTRUCTOR(SBCommandReturnObject,
                          (const lldb::SBCommandReturnObject &), rhs);

  m_opaque_up = clone(rhs.m_opaque_up);
}

SBCommandReturnObject &SBCommandReturnObject::
operator=(const SBCommandReturnObject &rhs) {
  LLDB_RECORD_METHOD(
      lldb::SBCommandReturnObject &,
      SBCommandReturnObject, operator=,(const lldb::SBCommandReturnObject &),
      rhs);

  if (this != &rhs)
    m_opaque_up = clone(rhs.m_opaque_up);
  return LLDB_RECORD_RESULT(*this);
}

SBCommandReturnObject::~SBCommandReturnObject() = default;

bool SBCommandReturnObject::IsValid() const {
  LLDB_RECORD_METHOD_CONST_NO_ARGS(bool, SBCommandReturnObject, IsValid);
  return this->operator bool();
}
SBCommandReturnObject::operator bool() const {
  LLDB_RECORD_METHOD_CONST_NO_ARGS(bool, SBCommandReturnObject, operator bool);

  // This method is not useful but it needs to stay to keep SB API stable.
  return true;
}

const char *SBCommandReturnObject::GetOutput() {
  LLDB_RECORD_METHOD_NO_ARGS(const char *, SBCommandReturnObject, GetOutput);

  ConstString output(ref().GetOutputData());
  return output.AsCString(/*value_if_empty*/ "");
}

const char *SBCommandReturnObject::GetError() {
  LLDB_RECORD_METHOD_NO_ARGS(const char *, SBCommandReturnObject, GetError);

  ConstString output(ref().GetErrorData());
  return output.AsCString(/*value_if_empty*/ "");
}

size_t SBCommandReturnObject::GetOutputSize() {
  LLDB_RECORD_METHOD_NO_ARGS(size_t, SBCommandReturnObject, GetOutputSize);

  return ref().GetOutputData().size();
}

size_t SBCommandReturnObject::GetErrorSize() {
  LLDB_RECORD_METHOD_NO_ARGS(size_t, SBCommandReturnObject, GetErrorSize);

  return ref().GetErrorData().size();
}

size_t SBCommandReturnObject::PutOutput(FILE *fh) {
  LLDB_RECORD_DUMMY(size_t, SBCommandReturnObject, PutOutput, (FILE *), fh);
  if (fh) {
    size_t num_bytes = GetOutputSize();
    if (num_bytes)
      return ::fprintf(fh, "%s", GetOutput());
  }
  return 0;
}

size_t SBCommandReturnObject::PutOutput(FileSP file_sp) {
  LLDB_RECORD_METHOD(size_t, SBCommandReturnObject, PutOutput, (FileSP),
                     file_sp);
  if (!file_sp)
    return 0;
  return file_sp->Printf("%s", GetOutput());
}

size_t SBCommandReturnObject::PutOutput(SBFile file) {
  LLDB_RECORD_METHOD(size_t, SBCommandReturnObject, PutOutput, (SBFile), file);
  if (!file.m_opaque_sp)
    return 0;
  return file.m_opaque_sp->Printf("%s", GetOutput());
}

size_t SBCommandReturnObject::PutError(FILE *fh) {
  LLDB_RECORD_DUMMY(size_t, SBCommandReturnObject, PutError, (FILE *), fh);
  if (fh) {
    size_t num_bytes = GetErrorSize();
    if (num_bytes)
      return ::fprintf(fh, "%s", GetError());
  }
  return 0;
}

size_t SBCommandReturnObject::PutError(FileSP file_sp) {
  LLDB_RECORD_METHOD(size_t, SBCommandReturnObject, PutError, (FileSP),
                     file_sp);
  if (!file_sp)
    return 0;
  return file_sp->Printf("%s", GetError());
}

size_t SBCommandReturnObject::PutError(SBFile file) {
  LLDB_RECORD_METHOD(size_t, SBCommandReturnObject, PutError, (SBFile), file);
  if (!file.m_opaque_sp)
    return 0;
  return file.m_opaque_sp->Printf("%s", GetError());
}

void SBCommandReturnObject::Clear() {
  LLDB_RECORD_METHOD_NO_ARGS(void, SBCommandReturnObject, Clear);

  ref().Clear();
}

lldb::ReturnStatus SBCommandReturnObject::GetStatus() {
  LLDB_RECORD_METHOD_NO_ARGS(lldb::ReturnStatus, SBCommandReturnObject,
                             GetStatus);

  return ref().GetStatus();
}

void SBCommandReturnObject::SetStatus(lldb::ReturnStatus status) {
  LLDB_RECORD_METHOD(void, SBCommandReturnObject, SetStatus,
                     (lldb::ReturnStatus), status);

  ref().SetStatus(status);
}

bool SBCommandReturnObject::Succeeded() {
  LLDB_RECORD_METHOD_NO_ARGS(bool, SBCommandReturnObject, Succeeded);

  return ref().Succeeded();
}

bool SBCommandReturnObject::HasResult() {
  LLDB_RECORD_METHOD_NO_ARGS(bool, SBCommandReturnObject, HasResult);

  return ref().HasResult();
}

void SBCommandReturnObject::AppendMessage(const char *message) {
  LLDB_RECORD_METHOD(void, SBCommandReturnObject, AppendMessage, (const char *),
                     message);

  ref().AppendMessage(message);
}

void SBCommandReturnObject::AppendWarning(const char *message) {
  LLDB_RECORD_METHOD(void, SBCommandReturnObject, AppendWarning, (const char *),
                     message);

  ref().AppendWarning(message);
}

CommandReturnObject *SBCommandReturnObject::operator->() const {
  return &**m_opaque_up;
}

CommandReturnObject *SBCommandReturnObject::get() const {
  return &**m_opaque_up;
}

CommandReturnObject &SBCommandReturnObject::operator*() const {
  return **m_opaque_up;
}

CommandReturnObject &SBCommandReturnObject::ref() const {
  return **m_opaque_up;
}

bool SBCommandReturnObject::GetDescription(SBStream &description) {
  LLDB_RECORD_METHOD(bool, SBCommandReturnObject, GetDescription,
                     (lldb::SBStream &), description);

  Stream &strm = description.ref();

  description.Printf("Error:  ");
  lldb::ReturnStatus status = ref().GetStatus();
  if (status == lldb::eReturnStatusStarted)
    strm.PutCString("Started");
  else if (status == lldb::eReturnStatusInvalid)
    strm.PutCString("Invalid");
  else if (ref().Succeeded())
    strm.PutCString("Success");
  else
    strm.PutCString("Fail");

  if (GetOutputSize() > 0)
    strm.Printf("\nOutput Message:\n%s", GetOutput());

  if (GetErrorSize() > 0)
    strm.Printf("\nError Message:\n%s", GetError());

  return true;
}

void SBCommandReturnObject::SetImmediateOutputFile(FILE *fh) {
  LLDB_RECORD_DUMMY(void, SBCommandReturnObject, SetImmediateOutputFile,
                    (FILE *), fh);

  SetImmediateOutputFile(fh, false);
}

void SBCommandReturnObject::SetImmediateErrorFile(FILE *fh) {
  LLDB_RECORD_DUMMY(void, SBCommandReturnObject, SetImmediateErrorFile,
                    (FILE *), fh);

  SetImmediateErrorFile(fh, false);
}

void SBCommandReturnObject::SetImmediateOutputFile(FILE *fh,
                                                   bool transfer_ownership) {
  LLDB_RECORD_DUMMY(void, SBCommandReturnObject, SetImmediateOutputFile,
                    (FILE *, bool), fh, transfer_ownership);
  FileSP file = std::make_shared<NativeFile>(fh, transfer_ownership);
  ref().SetImmediateOutputFile(file);
}

void SBCommandReturnObject::SetImmediateErrorFile(FILE *fh,
                                                  bool transfer_ownership) {
  LLDB_RECORD_DUMMY(void, SBCommandReturnObject, SetImmediateErrorFile,
                    (FILE *, bool), fh, transfer_ownership);
  FileSP file = std::make_shared<NativeFile>(fh, transfer_ownership);
  ref().SetImmediateErrorFile(file);
}

void SBCommandReturnObject::SetImmediateOutputFile(SBFile file) {
  LLDB_RECORD_METHOD(void, SBCommandReturnObject, SetImmediateOutputFile,
                     (SBFile), file);
  ref().SetImmediateOutputFile(file.m_opaque_sp);
}

void SBCommandReturnObject::SetImmediateErrorFile(SBFile file) {
  LLDB_RECORD_METHOD(void, SBCommandReturnObject, SetImmediateErrorFile,
                     (SBFile), file);
  ref().SetImmediateErrorFile(file.m_opaque_sp);
}

void SBCommandReturnObject::SetImmediateOutputFile(FileSP file_sp) {
  LLDB_RECORD_METHOD(void, SBCommandReturnObject, SetImmediateOutputFile,
                     (FileSP), file_sp);
  SetImmediateOutputFile(SBFile(file_sp));
}

void SBCommandReturnObject::SetImmediateErrorFile(FileSP file_sp) {
  LLDB_RECORD_METHOD(void, SBCommandReturnObject, SetImmediateErrorFile,
                     (FileSP), file_sp);
  SetImmediateErrorFile(SBFile(file_sp));
}

void SBCommandReturnObject::PutCString(const char *string, int len) {
  LLDB_RECORD_METHOD(void, SBCommandReturnObject, PutCString,
                     (const char *, int), string, len);

  if (len == 0 || string == nullptr || *string == 0) {
    return;
  } else if (len > 0) {
    std::string buffer(string, len);
    ref().AppendMessage(buffer.c_str());
  } else
    ref().AppendMessage(string);
}

const char *SBCommandReturnObject::GetOutput(bool only_if_no_immediate) {
  LLDB_RECORD_METHOD(const char *, SBCommandReturnObject, GetOutput, (bool),
                     only_if_no_immediate);

  if (!only_if_no_immediate ||
      ref().GetImmediateOutputStream().get() == nullptr)
    return GetOutput();
  return nullptr;
}

const char *SBCommandReturnObject::GetError(bool only_if_no_immediate) {
  LLDB_RECORD_METHOD(const char *, SBCommandReturnObject, GetError, (bool),
                     only_if_no_immediate);

  if (!only_if_no_immediate || ref().GetImmediateErrorStream().get() == nullptr)
    return GetError();
  return nullptr;
}

size_t SBCommandReturnObject::Printf(const char *format, ...) {
  va_list args;
  va_start(args, format);
  size_t result = ref().GetOutputStream().PrintfVarArg(format, args);
  va_end(args);
  return result;
}

void SBCommandReturnObject::SetError(lldb::SBError &error,
                                     const char *fallback_error_cstr) {
  LLDB_RECORD_METHOD(void, SBCommandReturnObject, SetError,
                     (lldb::SBError &, const char *), error,
                     fallback_error_cstr);

  if (error.IsValid())
    ref().SetError(error.ref(), fallback_error_cstr);
  else if (fallback_error_cstr)
    ref().SetError(Status(), fallback_error_cstr);
}

void SBCommandReturnObject::SetError(const char *error_cstr) {
  LLDB_RECORD_METHOD(void, SBCommandReturnObject, SetError, (const char *),
                     error_cstr);

  if (error_cstr)
    ref().SetError(error_cstr);
}

namespace lldb_private {
namespace repro {

template <>
void RegisterMethods<SBCommandReturnObject>(Registry &R) {
  LLDB_REGISTER_CONSTRUCTOR(SBCommandReturnObject, ());
  LLDB_REGISTER_CONSTRUCTOR(SBCommandReturnObject,
                            (lldb_private::CommandReturnObject &));
  LLDB_REGISTER_CONSTRUCTOR(SBCommandReturnObject,
                            (const lldb::SBCommandReturnObject &));
  LLDB_REGISTER_METHOD(
      lldb::SBCommandReturnObject &,
      SBCommandReturnObject, operator=,(const lldb::SBCommandReturnObject &));
  LLDB_REGISTER_METHOD_CONST(bool, SBCommandReturnObject, IsValid, ());
  LLDB_REGISTER_METHOD_CONST(bool, SBCommandReturnObject, operator bool, ());
  LLDB_REGISTER_METHOD(const char *, SBCommandReturnObject, GetOutput, ());
  LLDB_REGISTER_METHOD(const char *, SBCommandReturnObject, GetError, ());
  LLDB_REGISTER_METHOD(size_t, SBCommandReturnObject, GetOutputSize, ());
  LLDB_REGISTER_METHOD(size_t, SBCommandReturnObject, GetErrorSize, ());
  LLDB_REGISTER_METHOD(size_t, SBCommandReturnObject, PutOutput, (FILE *));
  LLDB_REGISTER_METHOD(size_t, SBCommandReturnObject, PutError, (FILE *));
  LLDB_REGISTER_METHOD(size_t, SBCommandReturnObject, PutOutput, (SBFile));
  LLDB_REGISTER_METHOD(size_t, SBCommandReturnObject, PutError, (SBFile));
  LLDB_REGISTER_METHOD(size_t, SBCommandReturnObject, PutOutput, (FileSP));
  LLDB_REGISTER_METHOD(size_t, SBCommandReturnObject, PutError, (FileSP));
  LLDB_REGISTER_METHOD(void, SBCommandReturnObject, Clear, ());
  LLDB_REGISTER_METHOD(lldb::ReturnStatus, SBCommandReturnObject, GetStatus,
                       ());
  LLDB_REGISTER_METHOD(void, SBCommandReturnObject, SetStatus,
                       (lldb::ReturnStatus));
  LLDB_REGISTER_METHOD(bool, SBCommandReturnObject, Succeeded, ());
  LLDB_REGISTER_METHOD(bool, SBCommandReturnObject, HasResult, ());
  LLDB_REGISTER_METHOD(void, SBCommandReturnObject, AppendMessage,
                       (const char *));
  LLDB_REGISTER_METHOD(void, SBCommandReturnObject, AppendWarning,
                       (const char *));
  LLDB_REGISTER_METHOD(bool, SBCommandReturnObject, GetDescription,
                       (lldb::SBStream &));
  LLDB_REGISTER_METHOD(void, SBCommandReturnObject, SetImmediateOutputFile,
                       (FILE *));
  LLDB_REGISTER_METHOD(void, SBCommandReturnObject, SetImmediateErrorFile,
                       (FILE *));
  LLDB_REGISTER_METHOD(void, SBCommandReturnObject, SetImmediateOutputFile,
                       (SBFile));
  LLDB_REGISTER_METHOD(void, SBCommandReturnObject, SetImmediateErrorFile,
                       (SBFile));
  LLDB_REGISTER_METHOD(void, SBCommandReturnObject, SetImmediateOutputFile,
                       (FileSP));
  LLDB_REGISTER_METHOD(void, SBCommandReturnObject, SetImmediateErrorFile,
                       (FileSP));
  LLDB_REGISTER_METHOD(void, SBCommandReturnObject, SetImmediateOutputFile,
                       (FILE *, bool));
  LLDB_REGISTER_METHOD(void, SBCommandReturnObject, SetImmediateErrorFile,
                       (FILE *, bool));
  LLDB_REGISTER_METHOD(void, SBCommandReturnObject, PutCString,
                       (const char *, int));
  LLDB_REGISTER_METHOD(const char *, SBCommandReturnObject, GetOutput,
                       (bool));
  LLDB_REGISTER_METHOD(const char *, SBCommandReturnObject, GetError, (bool));
  LLDB_REGISTER_METHOD(void, SBCommandReturnObject, SetError,
                       (lldb::SBError &, const char *));
  LLDB_REGISTER_METHOD(void, SBCommandReturnObject, SetError, (const char *));
}

}
}
