//===-- SWIG Interface for SBTrace.h ----------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

namespace lldb {

class LLDB_API SBTrace {
public:
  SBTrace();
  size_t GetTraceData(SBError &error, void *buf,
                      size_t size, size_t offset,
                      lldb::tid_t thread_id);

  size_t GetMetaData(SBError &error, void *buf,
                     size_t size, size_t offset,
                     lldb::tid_t thread_id);

  void StopTrace(SBError &error,
                 lldb::tid_t thread_id);

  void GetTraceConfig(SBTraceOptions &options,
                      SBError &error);

  lldb::user_id_t GetTraceUID();

  bool IsValid();

};
} // namespace lldb