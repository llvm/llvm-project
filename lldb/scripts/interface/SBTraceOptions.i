//===-- SWIG Interface for SBTraceOptions -----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

namespace lldb {

class LLDB_API SBTraceOptions {
public:
  SBTraceOptions();

  lldb::TraceType getType() const;

  uint64_t getTraceBufferSize() const;

  lldb::SBStructuredData getTraceParams(lldb::SBError &error);

  uint64_t getMetaDataBufferSize() const;

  void setTraceParams(lldb::SBStructuredData &params);

  void setType(lldb::TraceType type);

  void setTraceBufferSize(uint64_t size);

  void setMetaDataBufferSize(uint64_t size);

  void setThreadID(lldb::tid_t thread_id);

  lldb::tid_t getThreadID();

  bool IsValid();
};
}
