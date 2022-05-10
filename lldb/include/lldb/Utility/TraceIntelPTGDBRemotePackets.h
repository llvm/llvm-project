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

#include "llvm/Support/JSON.h"

#include <chrono>

/// See docs/lldb-gdb-remote.txt for more information.
namespace lldb_private {

// List of data kinds used by jLLDBGetState and jLLDBGetBinaryData.
struct IntelPTDataKinds {
  static const char *kProcFsCpuInfo;
  static const char *kTraceBuffer;
};

/// jLLDBTraceStart gdb-remote packet
/// \{
struct TraceIntelPTStartRequest : TraceStartRequest {
  /// Size in bytes to use for each thread's trace buffer.
  int64_t trace_buffer_size;

  /// Whether to enable TSC
  bool enable_tsc;

  /// PSB packet period
  llvm::Optional<int64_t> psb_period;

  /// Required when doing "process tracing".
  ///
  /// Limit in bytes on all the thread traces started by this "process trace"
  /// instance. When a thread is about to be traced and the limit would be hit,
  /// then a "tracing" stop event is triggered.
  llvm::Optional<int64_t> process_buffer_size_limit;

  /// Whether to have a trace buffer per thread or per cpu core.
  llvm::Optional<bool> per_core_tracing;
};

bool fromJSON(const llvm::json::Value &value, TraceIntelPTStartRequest &packet,
              llvm::json::Path path);

llvm::json::Value toJSON(const TraceIntelPTStartRequest &packet);
/// \}

/// jLLDBTraceGetState gdb-remote packet
/// \{

/// TSC to wall time conversion values defined in the Linux perf_event_open API
/// when the capibilities cap_user_time and cap_user_time_zero are set. See the
/// See the documentation of `time_zero` in
/// https://man7.org/linux/man-pages/man2/perf_event_open.2.html for more
/// information.
class LinuxPerfZeroTscConversion
    : public TraceCounterConversion<std::chrono::nanoseconds> {
public:
  /// Create new \a LinuxPerfZeroTscConversion object from the conversion values
  /// defined in the Linux perf_event_open API.
  LinuxPerfZeroTscConversion(uint32_t time_mult, uint16_t time_shift,
                             uint64_t time_zero)
      : m_time_mult(time_mult), m_time_shift(time_shift),
        m_time_zero(time_zero) {}

  /// Convert TSC value to nanosecond wall time. The beginning of time (0
  /// nanoseconds) is defined by the kernel at boot time and has no particularly
  /// useful meaning. On the other hand, this value is constant for an entire
  /// trace session.
  //  See 'time_zero' section of
  //  https://man7.org/linux/man-pages/man2/perf_event_open.2.html
  ///
  /// \param[in] tsc
  ///   The TSC value to be converted.
  ///
  /// \return
  ///   Nanosecond wall time.
  std::chrono::nanoseconds Convert(uint64_t raw_counter_value) override;

  llvm::json::Value toJSON() override;

private:
  uint32_t m_time_mult;
  uint16_t m_time_shift;
  uint64_t m_time_zero;
};

struct TraceIntelPTGetStateResponse : TraceGetStateResponse {
  /// The TSC to wall time conversion if it exists, otherwise \b nullptr.
  TraceTscConversionUP tsc_conversion;
};

bool fromJSON(const llvm::json::Value &value,
              TraceTscConversionUP &tsc_conversion, llvm::json::Path path);

bool fromJSON(const llvm::json::Value &value,
              TraceIntelPTGetStateResponse &packet, llvm::json::Path path);

llvm::json::Value toJSON(const TraceIntelPTGetStateResponse &packet);
/// \}

} // namespace lldb_private

#endif // LLDB_UTILITY_TRACEINTELPTGDBREMOTEPACKETS_H
