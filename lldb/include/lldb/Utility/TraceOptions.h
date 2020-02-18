//===-- TraceOptions.h ------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_UTILITY_TRACEOPTIONS_H
#define LLDB_UTILITY_TRACEOPTIONS_H

#include "lldb/lldb-defines.h"
#include "lldb/lldb-enumerations.h"

#include "lldb/Utility/StructuredData.h"

namespace lldb_private {
class TraceOptions {
public:
  TraceOptions() : m_trace_params(new StructuredData::Dictionary()) {}

  const StructuredData::DictionarySP &getTraceParams() const {
    return m_trace_params;
  }

  lldb::TraceType getType() const { return m_type; }

  uint64_t getTraceBufferSize() const { return m_trace_buffer_size; }

  uint64_t getMetaDataBufferSize() const { return m_meta_data_buffer_size; }

  void setTraceParams(const StructuredData::DictionarySP &dict_obj) {
    m_trace_params = dict_obj;
  }

  void setType(lldb::TraceType type) { m_type = type; }

  void setTraceBufferSize(uint64_t size) { m_trace_buffer_size = size; }

  void setMetaDataBufferSize(uint64_t size) { m_meta_data_buffer_size = size; }

  void setThreadID(lldb::tid_t thread_id) { m_thread_id = thread_id; }

  lldb::tid_t getThreadID() const { return m_thread_id; }

private:
  lldb::TraceType m_type;
  uint64_t m_trace_buffer_size;
  uint64_t m_meta_data_buffer_size;
  lldb::tid_t m_thread_id;

  /// m_trace_params is meant to hold any custom parameters
  /// apart from meta buffer size and trace size.
  /// The interpretation of such parameters is left to
  /// the lldb-server.
  StructuredData::DictionarySP m_trace_params;
};
}

#endif // LLDB_UTILITY_TRACEOPTIONS_H
