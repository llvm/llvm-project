//===-- SBTrace.cpp -------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "SBReproducerPrivate.h"
#include "lldb/Target/Process.h"

#include "lldb/API/SBStructuredData.h"
#include "lldb/API/SBThread.h"
#include "lldb/API/SBTrace.h"

#include "lldb/Core/StructuredDataImpl.h"

#include <memory>

using namespace lldb;
using namespace lldb_private;

SBTrace::SBTrace() { LLDB_RECORD_CONSTRUCTOR_NO_ARGS(SBTrace); }

SBTrace::SBTrace(const lldb::TraceSP &trace_sp) : m_opaque_sp(trace_sp) {
  LLDB_RECORD_CONSTRUCTOR(SBTrace, (const lldb::TraceSP &), trace_sp);
}

const char *SBTrace::GetStartConfigurationHelp() {
  LLDB_RECORD_METHOD_NO_ARGS(const char *, SBTrace, GetStartConfigurationHelp);
  return LLDB_RECORD_RESULT(
      m_opaque_sp ? m_opaque_sp->GetStartConfigurationHelp() : nullptr);
}

SBError SBTrace::Start(const SBStructuredData &configuration) {
  LLDB_RECORD_METHOD(SBError, SBTrace, Start, (const SBStructuredData &),
                     configuration);
  SBError error;
  if (!m_opaque_sp)
    error.SetErrorString("error: invalid trace");
  else if (llvm::Error err =
               m_opaque_sp->Start(configuration.m_impl_up->GetObjectSP()))
    error.SetErrorString(llvm::toString(std::move(err)).c_str());
  return LLDB_RECORD_RESULT(error);
}

SBError SBTrace::Start(const SBThread &thread,
                       const SBStructuredData &configuration) {
  LLDB_RECORD_METHOD(SBError, SBTrace, Start,
                     (const SBThread &, const SBStructuredData &), thread,
                     configuration);

  SBError error;
  if (!m_opaque_sp)
    error.SetErrorString("error: invalid trace");
  else {
    if (llvm::Error err =
            m_opaque_sp->Start(std::vector<lldb::tid_t>{thread.GetThreadID()},
                               configuration.m_impl_up->GetObjectSP()))
      error.SetErrorString(llvm::toString(std::move(err)).c_str());
  }

  return LLDB_RECORD_RESULT(error);
}

SBError SBTrace::Stop() {
  LLDB_RECORD_METHOD_NO_ARGS(SBError, SBTrace, Stop);
  SBError error;
  if (!m_opaque_sp)
    error.SetErrorString("error: invalid trace");
  else if (llvm::Error err = m_opaque_sp->Stop())
    error.SetErrorString(llvm::toString(std::move(err)).c_str());
  return LLDB_RECORD_RESULT(error);
}

SBError SBTrace::Stop(const SBThread &thread) {
  LLDB_RECORD_METHOD(SBError, SBTrace, Stop, (const SBThread &), thread);
  SBError error;
  if (!m_opaque_sp)
    error.SetErrorString("error: invalid trace");
  else if (llvm::Error err = m_opaque_sp->Stop({thread.GetThreadID()}))
    error.SetErrorString(llvm::toString(std::move(err)).c_str());
  return LLDB_RECORD_RESULT(error);
}

bool SBTrace::IsValid() {
  LLDB_RECORD_METHOD_NO_ARGS(bool, SBTrace, IsValid);
  return this->operator bool();
}

SBTrace::operator bool() const {
  LLDB_RECORD_METHOD_CONST_NO_ARGS(bool, SBTrace, operator bool);
  return (bool)m_opaque_sp;
}

namespace lldb_private {
namespace repro {

template <>
void RegisterMethods<SBTrace>(Registry &R) {
  LLDB_REGISTER_CONSTRUCTOR(SBTrace, ());
  LLDB_REGISTER_CONSTRUCTOR(SBTrace, (const lldb::TraceSP &));
  LLDB_REGISTER_METHOD(SBError, SBTrace, Start, (const SBStructuredData &));
  LLDB_REGISTER_METHOD(SBError, SBTrace, Start,
                       (const SBThread &, const SBStructuredData &));
  LLDB_REGISTER_METHOD(SBError, SBTrace, Stop, (const SBThread &));
  LLDB_REGISTER_METHOD(SBError, SBTrace, Stop, ());
  LLDB_REGISTER_METHOD(bool, SBTrace, IsValid, ());
  LLDB_REGISTER_METHOD(const char *, SBTrace, GetStartConfigurationHelp, ());
  LLDB_REGISTER_METHOD_CONST(bool, SBTrace, operator bool, ());
}

}
}
