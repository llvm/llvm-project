//===-- TraceIntelPTGDBRemotePackets.h --------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_UTILITY_TRACEINTELPTGDBREMOTEPACKETS_H
#define LLDB_UTILITY_TRACEINTELPTGDBREMOTEPACKETS_H

#include "lldb/Utility/TraceGDBRemotePackets.h"

/// See docs/lldb-gdb-remote.txt for more information.
namespace lldb_private {

/// jLLDBTraceStart gdb-remote packet
/// \{
struct TraceIntelPTStartRequest : TraceStartRequest {
  /// Size in bytes to use for each thread's trace buffer.
  int64_t threadBufferSize;
  /// Required when doing "process tracing".
  ///
  /// Limit in bytes on all the thread traces started by this "process trace"
  /// instance. When a thread is about to be traced and the limit would be hit,
  /// then a "tracing" stop event is triggered.
  llvm::Optional<int64_t> processBufferSizeLimit;
};

bool fromJSON(const llvm::json::Value &value, TraceIntelPTStartRequest &packet,
              llvm::json::Path path);

llvm::json::Value toJSON(const TraceIntelPTStartRequest &packet);
/// \}

} // namespace lldb_private

#endif // LLDB_UTILITY_TRACEINTELPTGDBREMOTEPACKETS_H
