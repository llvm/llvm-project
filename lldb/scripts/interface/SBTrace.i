//===-- SWIG Interface for SBTrace.h ----------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
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