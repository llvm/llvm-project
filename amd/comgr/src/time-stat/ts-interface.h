//===- ts-interface.h - Timing statistics ---------------------------------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef AMD_COMGR_TS_INTERFACE_H
#define AMD_COMGR_TS_INTERFACE_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"

#include "amd_comgr.h" // for amd_comgr_action_kind_t
#include <cstdint>
// External interface

namespace COMGR {
namespace TimeStatistics {

struct ProfilePoint {
  ProfilePoint(llvm::StringRef Name);
  ~ProfilePoint();
  void finish();

private:
  std::string Name = "";
  double StartTime = 0.0;
  bool isFinished = false;
};

// A pre-aggregated statistics row folded into the shared PerfStats in one shot,
// so a producer can accumulate locally and pay the global lock only once.
struct PerfStatRecord {
  llvm::StringRef Name;
  double TimeTaken = 0.0; // already in the configured granularity units
  uint64_t Calls = 0;
  uint64_t Patches = 0;
  double MinTime = 0.0;
  double MaxTime = 0.0;
};

bool InitTimeStatistics(std::string LogFile);
void StartAction(amd_comgr_action_kind_t);
void EndAction();

/// Thread-safe batch merge of \p Records into the process-wide PerfStats under
/// a single lock. Lazily initializes the sink; a no-op when time statistics are
/// not enabled (AMD_COMGR_TIME_STATISTICS unset).
void mergeStats(llvm::ArrayRef<PerfStatRecord> Records);

} // namespace TimeStatistics
} // namespace COMGR

#endif // AMD_COMGR_TS_INTERFACE_H
