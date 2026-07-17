//===- time-stat.cpp - Timing statistics ----------------------------------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file implements Comgr's built-in profiler, which can be enabled with
/// the AMD_COMGR_TIME_STATISTICS environment variable.
///
//===----------------------------------------------------------------------===//

#include <functional>
#include <iostream>
#include <memory>
#include <mutex>
#include <stdlib.h>
#include <system_error>

#include "comgr-env.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/raw_ostream.h"

#if defined _WIN64 || defined _WIN32
// Avoid introducing min as a macro from Windows headers.
#define NOMINMAX
#include <windows.h>
#else
#include <time.h>
#endif

#if defined(__FreeBSD__) && !defined(CLOCK_MONOTONIC_RAW)
#define CLOCK_MONOTONIC_RAW CLOCK_MONOTONIC
#endif

using namespace llvm;
using namespace COMGR;

#include "time-stat.h"
#include "ts-interface.h"

namespace COMGR {
namespace TimeStatistics {

namespace {
std::unique_ptr<PerfStats> PS = nullptr;
void dump() {
  PS->dumpPerfStats();
  PS.reset();
}
} // namespace

void getLogFile(std::string &PerfLog) {
  if (std::optional<StringRef> RedirectLogs = env::getRedirectLogs()) {
    PerfLog = (*RedirectLogs).str();
    return;
  }
  PerfLog = "PerfStatsLog.txt";
}

bool InitTimeStatistics(std::string LogFile) {
  // Thread-safe lazy init: call_once creates PS and registers the atexit dump
  // exactly once, even if concurrent Comgr calls reach here simultaneously.
  static std::once_flag InitFlag;
  std::call_once(InitFlag, [&LogFile]() {
    if (!env::needTimeStatistics())
      return;

    if (LogFile == "")
      getLogFile(LogFile);

    std::unique_ptr<PerfStats> Stats = std::make_unique<PerfStats>();
    if (!Stats->Init(LogFile)) {
      std::cerr << "TimeStatistics failed to initialize\n";
      return;
    }
    PS = std::move(Stats);
    std::atexit(&dump);
  });
  return PS != nullptr;
}

void ProfilePoint::finish() {
  if (PS) {
    double End = PS->getCurrentTime();
    PS->AddToStats(Name, End - StartTime);
  }

  isFinished = true;
}

ProfilePoint::ProfilePoint(StringRef Tag) : Name(Tag) {
  InitTimeStatistics("");
  if (PS) {
    StartTime = PS->getCurrentTime();
  }
}

ProfilePoint::~ProfilePoint() {
  if (!isFinished) {
    finish();
  }
}

void mergeStats(llvm::ArrayRef<PerfStatRecord> Records) {
  // Lazily stand up the sink (and the atexit dump) exactly as ProfilePoint
  // does; a no-op when AMD_COMGR_TIME_STATISTICS is unset.
  if (!InitTimeStatistics(""))
    return;
  if (PS)
    PS->mergeStats(Records);
}

// Timer implementation
#if defined _WIN64 || defined _WIN32
class PerfTimerWindows : public PerfTimerImpl {

public:
  PerfTimerWindows() {};
  virtual bool Init() override {
    LARGE_INTEGER li;
    if (QueryPerformanceCounter(&li))
      CounterStart = li.QuadPart;
    else {
      std::cerr << "Failed to get performance counter\n";
      return false;
    }

    if (!QueryPerformanceFrequency(&li)) {
      std::cerr << "Failed to get performance frequency\n";
      return false;
    }
    // QueryPerformanceFrequency returns counts per second
    // If we need milliseconds we divide by 10^3
    GranularityPerSecond = env::getGranularityUnitsPerSecond();
    PCFreq = li.QuadPart / GranularityPerSecond;
    return true;
  }

  virtual double getCurrentTime() override {
    LARGE_INTEGER li;
    if (QueryPerformanceCounter(&li))
      return double(li.QuadPart) / PCFreq;
    else {
      std::cerr << "Failed to get performance counter\n";
      return 0.0;
    }
  }
};

#else
class PerfTimerPosix : public PerfTimerImpl {
public:
  virtual bool Init() override {
    struct timespec StartTime;
    if (!clock_gettime(CLOCK_MONOTONIC_RAW, &StartTime)) {
      CounterStart = StartTime.tv_sec * 1e9 + StartTime.tv_nsec;
    } else {
      std::cerr << "Failed to get performance counter\n";
      return false;
    }

    struct timespec Res;
    if (clock_getres(CLOCK_MONOTONIC_RAW, &Res)) {
      std::cerr << "Failed to get performance frequency\n";
      return false;
    }
    // clock_getres returns counts per nanosecond
    // If we need milliseconds we multiply by 10^6
    GranularityPerSecond = env::getGranularityUnitsPerSecond();
    PCFreq = (Res.tv_sec * 1e9 + Res.tv_nsec) * (1e9 / GranularityPerSecond);
    return true;
  }

  virtual double getCurrentTime() override {
    struct timespec EndTime;
    if (!clock_gettime(CLOCK_MONOTONIC_RAW, &EndTime)) {
      return (EndTime.tv_sec * 1e9 + EndTime.tv_nsec) / PCFreq;
    }
    std::cerr << "Failed to get performance counter\n";
    return 0.0;
  }
};
#endif

bool PerfTimer::Init() {
#if defined _WIN64 || defined _WIN32
  pImpl = std::make_unique<PerfTimerWindows>();
#else
  pImpl = std::make_unique<PerfTimerPosix>();
#endif
  return pImpl->Init();
}

} // namespace TimeStatistics
} // namespace COMGR
