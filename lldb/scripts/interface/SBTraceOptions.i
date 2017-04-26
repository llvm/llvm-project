//===-- SWIG Interface for SBTraceOptions -----------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
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
