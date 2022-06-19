//===-- SBTrace.cpp -------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Target/Process.h"
#include "lldb/Utility/Instrumentation.h"

#include "lldb/API/SBStructuredData.h"
#include "lldb/API/SBThread.h"
#include "lldb/API/SBTrace.h"

#include "lldb/Core/StructuredDataImpl.h"

#include <memory>

using namespace lldb;
using namespace lldb_private;

SBTrace::SBTrace() { LLDB_INSTRUMENT_VA(this); }

SBTrace::SBTrace(const lldb::TraceSP &trace_sp) : m_opaque_sp(trace_sp) {
  LLDB_INSTRUMENT_VA(this, trace_sp);
}

const char *SBTrace::GetStartConfigurationHelp() {
  LLDB_INSTRUMENT_VA(this);
  return m_opaque_sp ? m_opaque_sp->GetStartConfigurationHelp() : nullptr;
}

SBError SBTrace::Start(const SBStructuredData &configuration) {
  LLDB_INSTRUMENT_VA(this, configuration);
  SBError error;
  if (!m_opaque_sp)
    error.SetErrorString("error: invalid trace");
  else if (llvm::Error err =
               m_opaque_sp->Start(configuration.m_impl_up->GetObjectSP()))
    error.SetErrorString(llvm::toString(std::move(err)).c_str());
  return error;
}

SBError SBTrace::Start(const SBThread &thread,
                       const SBStructuredData &configuration) {
  LLDB_INSTRUMENT_VA(this, thread, configuration);

  SBError error;
  if (!m_opaque_sp)
    error.SetErrorString("error: invalid trace");
  else {
    if (llvm::Error err =
            m_opaque_sp->Start(std::vector<lldb::tid_t>{thread.GetThreadID()},
                               configuration.m_impl_up->GetObjectSP()))
      error.SetErrorString(llvm::toString(std::move(err)).c_str());
  }

  return error;
}

SBError SBTrace::Stop() {
  LLDB_INSTRUMENT_VA(this);
  SBError error;
  if (!m_opaque_sp)
    error.SetErrorString("error: invalid trace");
  else if (llvm::Error err = m_opaque_sp->Stop())
    error.SetErrorString(llvm::toString(std::move(err)).c_str());
  return error;
}

SBError SBTrace::Stop(const SBThread &thread) {
  LLDB_INSTRUMENT_VA(this, thread);
  SBError error;
  if (!m_opaque_sp)
    error.SetErrorString("error: invalid trace");
  else if (llvm::Error err = m_opaque_sp->Stop({thread.GetThreadID()}))
    error.SetErrorString(llvm::toString(std::move(err)).c_str());
  return error;
}

bool SBTrace::IsValid() {
  LLDB_INSTRUMENT_VA(this);
  return this->operator bool();
}

SBTrace::operator bool() const {
  LLDB_INSTRUMENT_VA(this);
  return (bool)m_opaque_sp;
}
