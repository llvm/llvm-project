//===- time-stat.h - Timing statistics ------------------------------------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef AMD_COMGR_TIME_STAT_H
#define AMD_COMGR_TIME_STAT_H

#include "perf-timer.h"
#include "ts-interface.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/raw_ostream.h"

#include "amd_comgr.h"
#include <algorithm>
#include <functional>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>

namespace COMGR {
namespace TimeStatistics {

struct ProfileData {
  double TimeTaken = 0.0;
  int Counter = 0;
  // Extra columns produced by batch-merged callers (mergeStats). Defaulted so
  // ProfilePoint-based callers, which only record wall time, are unaffected.
  uint64_t Patches = 0;
  double MinTime = std::numeric_limits<double>::max();
  double MaxTime = 0.0;
};

class PerfStats {
  std::unique_ptr<llvm::raw_fd_ostream,
                  std::function<void(llvm::raw_fd_ostream *)>>
      pLog;
  PerfTimer PT;

  // Guards ProfileDataMap so AddToStats / mergeStats / dumpPerfStats are safe
  // under concurrent Comgr calls (e.g. concurrent hotswap rewrites).
  std::mutex Mtx;
  llvm::StringMap<ProfileData> ProfileDataMap;

public:
  PerfStats() {}
  bool Init(std::string LogFile) {
    std::error_code EC;
    std::unique_ptr<llvm::raw_fd_ostream> LogF(
        new (std::nothrow)
            llvm::raw_fd_ostream(LogFile, EC, llvm::sys::fs::OF_Text));
    if (EC) {
      std::cerr << "Failed to open log file " << LogFile << "for perf stats "
                << EC.message() << "\n ";
      return false;
    } else {
      pLog = std::move(LogF);
    }

    // Initialize Timer
    if (!PT.Init())
      return false;

    return true;
  }

  double getCurrentTime() { return PT.getCurrentTime(); }

  void AddToStats(llvm::StringRef Name, double TimeTaken) {
    std::scoped_lock Lock(Mtx);
    ProfileData &D = ProfileDataMap[Name];
    D.TimeTaken += TimeTaken;
    D.Counter++;
    D.MinTime = std::min(D.MinTime, TimeTaken);
    D.MaxTime = std::max(D.MaxTime, TimeTaken);
  }

  // Fold many pre-aggregated rows in under a single lock so hot-path producers
  // can accumulate lock-free and merge once.
  void mergeStats(llvm::ArrayRef<PerfStatRecord> Records) {
    std::scoped_lock Lock(Mtx);
    for (const PerfStatRecord &R : Records) {
      ProfileData &D = ProfileDataMap[R.Name];
      D.TimeTaken += R.TimeTaken;
      D.Counter += static_cast<int>(R.Calls);
      D.Patches += R.Patches;
      if (R.Calls) {
        D.MinTime = std::min(D.MinTime, R.MinTime);
        D.MaxTime = std::max(D.MaxTime, R.MaxTime);
      }
    }
  }

  void dumpPerfStats() {
    std::scoped_lock Lock(Mtx);
    llvm::StringRef Unit = env::getTimeStatisticsGranularity();
    for (const auto &Item : ProfileDataMap) {
      const ProfileData &D = Item.getValue();
      const double MinT =
          D.MinTime == std::numeric_limits<double>::max() ? 0.0 : D.MinTime;
      // Keep the granularity unit last so existing consumers that anchor on it
      // (e.g. the time-statistics.cl "grep 'ms$'" check) still match; the extra
      // columns are inserted before it.
      *pLog << llvm::format("%-50s", Item.getKey().str().c_str())
            << llvm::format("%6d", D.Counter) << " calls "
            << llvm::format("%10.4f", D.TimeTaken)
            << llvm::format("  min %10.4f  max %10.4f  patches %6llu", MinT,
                            D.MaxTime,
                            static_cast<unsigned long long>(D.Patches))
            << " " << Unit << "\n";
    }
  }

  // Snapshot an aggregated row by name for unit tests; returns a zeroed
  // ProfileData when the row is absent. Locks like the other accessors.
  ProfileData lookupForTest(llvm::StringRef Name) {
    std::scoped_lock Lock(Mtx);
    return ProfileDataMap.lookup(Name);
  }
};

} // namespace TimeStatistics
} // namespace COMGR

#endif // AMD_COMGR_TIME_STAT_H
