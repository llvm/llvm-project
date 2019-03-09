//===-- SBTrace.cpp ---------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "SBReproducerPrivate.h"
#include "lldb/Target/Process.h"

#include "lldb/API/SBTrace.h"
#include "lldb/API/SBTraceOptions.h"

#include <memory>

using namespace lldb;
using namespace lldb_private;

class TraceImpl {
public:
  lldb::user_id_t uid;
};

lldb::ProcessSP SBTrace::GetSP() const { return m_opaque_wp.lock(); }

size_t SBTrace::GetTraceData(SBError &error, void *buf, size_t size,
                             size_t offset, lldb::tid_t thread_id) {
  LLDB_RECORD_DUMMY(size_t, SBTrace, GetTraceData,
                    (lldb::SBError &, void *, size_t, size_t, lldb::tid_t),
                    error, buf, size, offset, thread_id);

  ProcessSP process_sp(GetSP());
  llvm::MutableArrayRef<uint8_t> buffer(static_cast<uint8_t *>(buf), size);
  error.Clear();

  if (!process_sp) {
    error.SetErrorString("invalid process");
  } else {
    error.SetError(
        process_sp->GetData(GetTraceUID(), thread_id, buffer, offset));
  }
  return buffer.size();
}

size_t SBTrace::GetMetaData(SBError &error, void *buf, size_t size,
                            size_t offset, lldb::tid_t thread_id) {
  LLDB_RECORD_DUMMY(size_t, SBTrace, GetMetaData,
                    (lldb::SBError &, void *, size_t, size_t, lldb::tid_t),
                    error, buf, size, offset, thread_id);

  ProcessSP process_sp(GetSP());
  llvm::MutableArrayRef<uint8_t> buffer(static_cast<uint8_t *>(buf), size);
  error.Clear();

  if (!process_sp) {
    error.SetErrorString("invalid process");
  } else {
    error.SetError(
        process_sp->GetMetaData(GetTraceUID(), thread_id, buffer, offset));
  }
  return buffer.size();
}

void SBTrace::StopTrace(SBError &error, lldb::tid_t thread_id) {
  LLDB_RECORD_METHOD(void, SBTrace, StopTrace, (lldb::SBError &, lldb::tid_t),
                     error, thread_id);

  ProcessSP process_sp(GetSP());
  error.Clear();

  if (!process_sp) {
    error.SetErrorString("invalid process");
    return;
  }
  error.SetError(process_sp->StopTrace(GetTraceUID(), thread_id));
}

void SBTrace::GetTraceConfig(SBTraceOptions &options, SBError &error) {
  LLDB_RECORD_METHOD(void, SBTrace, GetTraceConfig,
                     (lldb::SBTraceOptions &, lldb::SBError &), options, error);

  ProcessSP process_sp(GetSP());
  error.Clear();

  if (!process_sp) {
    error.SetErrorString("invalid process");
  } else {
    error.SetError(process_sp->GetTraceConfig(GetTraceUID(),
                                              *(options.m_traceoptions_sp)));
  }
}

lldb::user_id_t SBTrace::GetTraceUID() {
  LLDB_RECORD_METHOD_NO_ARGS(lldb::user_id_t, SBTrace, GetTraceUID);

  if (m_trace_impl_sp)
    return m_trace_impl_sp->uid;
  return LLDB_INVALID_UID;
}

void SBTrace::SetTraceUID(lldb::user_id_t uid) {
  if (m_trace_impl_sp)
    m_trace_impl_sp->uid = uid;
}

SBTrace::SBTrace() {
  LLDB_RECORD_CONSTRUCTOR_NO_ARGS(SBTrace);

  m_trace_impl_sp = std::make_shared<TraceImpl>();
  if (m_trace_impl_sp)
    m_trace_impl_sp->uid = LLDB_INVALID_UID;
}

void SBTrace::SetSP(const ProcessSP &process_sp) { m_opaque_wp = process_sp; }

bool SBTrace::IsValid() {
  LLDB_RECORD_METHOD_NO_ARGS(bool, SBTrace, IsValid);

  if (!m_trace_impl_sp)
    return false;
  if (!GetSP())
    return false;
  return true;
}
