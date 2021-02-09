#include <functional>
#include <iostream>
#include <memory>
#include <stdlib.h>
#include <system_error>

#include "llvm/ADT/Optional.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/raw_ostream.h"

#if defined _WIN64 || defined _WIN32
#include <windows.h>
#elif defined __linux__
#include <time.h>
#endif

using namespace llvm;

#include "time-stat.h"
#include "ts-interface.h"

namespace TimeStatistics {

static std::unique_ptr<PerfStats> PS = nullptr;
static void dump() { PS->dumpPerfStats(); }

bool InitTimeStatistics(std::string LogFile) {
  if (!PS) {
    PS = std::make_unique<PerfStats>();
    if (!PS || !PS->Init(LogFile)) {
      std::cerr << "TimeStatistics failed to initialize\n";
      return false;
    }
    std::atexit(&dump);
  }
  return true;
}

void StartAction(amd_comgr_action_kind_t ActionKind) {
  if (PS) {
    PS->StartAction(ActionKind);
  }
}

void EndAction() {
  if (PS) {
    PS->WriteActionTime();
  }
}

// Timer implementation
#if defined _WIN64 || defined _WIN32
class PerfTimerWindows : public PerfTimerImpl {

public:
  PerfTimerWindows(){};
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
    // TODO: granularity as env var
    PCFreq = li.QuadPart / 1e3;
    return true;
  }
  virtual double getTimer() override {
    LARGE_INTEGER li;
    if (QueryPerformanceCounter(&li))
      return double(li.QuadPart - CounterStart) / PCFreq;
    else {
      std::cerr << "Failed to get performance counter\n";
      return 0.0;
    }
  }
};

#elif defined __linux__
class PerfTimerLinux : public PerfTimerImpl {
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
    if (!clock_getres(CLOCK_MONOTONIC_RAW, &Res)) {
      std::cerr << "Failed to get performance frequency\n";
      return false;
    }
    // clock_getres returns counts per nanosecond
    // If we need milliseconds we multiply by 10^3
    // TODO: granularity as env var
    PCFreq = (Res.tv_sec * 1e9 + Res.tv_nsec) * 1e3;
    return true;
  }
  virtual double getTimer() override {
    struct timespec EndTime;
    if (!clock_gettime(CLOCK_MONOTONIC_RAW, &EndTime)) {
      return ((EndTime.tv_sec * 1e9 + EndTime.tv_nsec) - CounterStart) / PCFreq;
    }
    std::cerr << "Failed to get performance counter\n";
    return 0.0;
  }
};
#endif

bool PerfTimer::Init() {
#if defined _WIN64 || defined _WIN32
  pImpl = std::make_unique<PerfTimerWindows>();
#elif defined __linux__
  pImpl = std::make_unique<PerfTimerLinux>();
#endif
  return pImpl->Init();
}

} // namespace TimeStatistics
