// Copyright 2015 Google Inc. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "benchmark/macros.h"
#include "internal_macros.h"
#include "walltime.h"

#if defined(BENCHMARK_OS_WINDOWS)
#include <time.h>
#include <winsock.h> // for timeval
#else
#include <sys/time.h>
#endif

#include <cstdio>
#include <cstdint>
#include <cstring>
#include <ctime>

#include <atomic>
#include <chrono>
#include <limits>

#include "arraysize.h"
#include "check.h"
#include "cycleclock.h"
#include "log.h"
#include "sysinfo.h"

namespace benchmark {
namespace walltime {

namespace {

#if defined(HAVE_STEADY_CLOCK)
template <bool HighResIsSteady = std::chrono::high_resolution_clock::is_steady>
struct ChooseSteadyClock {
    typedef std::chrono::high_resolution_clock type;
};

template <>
struct ChooseSteadyClock<false> {
    typedef std::chrono::steady_clock type;
};
#endif

struct ChooseClockType {
#if defined(HAVE_STEADY_CLOCK)
  typedef ChooseSteadyClock<>::type type;
#else
  typedef std::chrono::high_resolution_clock type;
#endif
};

class WallTimeImp
{
public:
  WallTime Now();

  static WallTimeImp& GetWallTimeImp() {
    static WallTimeImp* imp = new WallTimeImp();
    return *imp;
  }

private:
  WallTimeImp();
  // Helper routines to load/store a float from an AtomicWord. Required because
  // g++ < 4.7 doesn't support std::atomic<float> correctly. I cannot wait to
  // get rid of this horror show.
  void SetDrift(float f) {
    int32_t w;
    memcpy(&w, &f, sizeof(f));
    std::atomic_store(&drift_adjust_, w);
  }

  float GetDrift() const {
    float f;
    int32_t w = std::atomic_load(&drift_adjust_);
    memcpy(&f, &w, sizeof(f));
    return f;
  }

  WallTime Slow() const {
    struct timeval tv;
#if defined(BENCHMARK_OS_WINDOWS)
    FILETIME    file_time;
    SYSTEMTIME  system_time;
    ULARGE_INTEGER ularge;
    const unsigned __int64 epoch = 116444736000000000LL;

    GetSystemTime(&system_time);
    SystemTimeToFileTime(&system_time, &file_time);
    ularge.LowPart = file_time.dwLowDateTime;
    ularge.HighPart = file_time.dwHighDateTime;

    tv.tv_sec = (long)((ularge.QuadPart - epoch) / (10L * 1000 * 1000));
    tv.tv_usec = (long)(system_time.wMilliseconds * 1000);
#else
    gettimeofday(&tv, nullptr);
#endif
    return tv.tv_sec + tv.tv_usec * 1e-6;
  }

private:
  static_assert(sizeof(float) <= sizeof(int32_t),
               "type sizes don't allow the drift_adjust hack");

  WallTime base_walltime_;
  int64_t base_cycletime_;
  int64_t cycles_per_second_;
  double seconds_per_cycle_;
  uint32_t last_adjust_time_;
  std::atomic<int32_t> drift_adjust_;
  int64_t max_interval_cycles_;

  BENCHMARK_DISALLOW_COPY_AND_ASSIGN(WallTimeImp);
};


WallTime WallTimeImp::Now() {
  WallTime now = 0.0;
  WallTime result = 0.0;
  int64_t ct = 0;
  uint32_t top_bits = 0;
  do {
    ct = cycleclock::Now();
    int64_t cycle_delta = ct - base_cycletime_;
    result = base_walltime_ + cycle_delta * seconds_per_cycle_;

    top_bits = static_cast<uint32_t>(uint64_t(ct) >> 32);
    // Recompute drift no more often than every 2^32 cycles.
    // I.e., @2GHz, ~ every two seconds
    if (top_bits == last_adjust_time_) {  // don't need to recompute drift
      return result + GetDrift();
    }

    now = Slow();
  } while (cycleclock::Now() - ct > max_interval_cycles_);
  // We are now sure that "now" and "result" were produced within
  // kMaxErrorInterval of one another.

  SetDrift(static_cast<float>(now - result));
  last_adjust_time_ = top_bits;
  return now;
}


WallTimeImp::WallTimeImp()
    : base_walltime_(0.0), base_cycletime_(0),
      cycles_per_second_(0), seconds_per_cycle_(0.0),
      last_adjust_time_(0), drift_adjust_(0),
      max_interval_cycles_(0) {
  const double kMaxErrorInterval = 100e-6;
  cycles_per_second_ = static_cast<int64_t>(CyclesPerSecond());
  CHECK(cycles_per_second_ != 0);
  seconds_per_cycle_ = 1.0 / cycles_per_second_;
  max_interval_cycles_ =
      static_cast<int64_t>(cycles_per_second_ * kMaxErrorInterval);
  do {
    base_cycletime_ = cycleclock::Now();
    base_walltime_ = Slow();
  } while (cycleclock::Now() - base_cycletime_ > max_interval_cycles_);
  // We are now sure that "base_walltime" and "base_cycletime" were produced
  // within kMaxErrorInterval of one another.

  SetDrift(0.0);
  last_adjust_time_ = static_cast<uint32_t>(uint64_t(base_cycletime_) >> 32);
}

WallTime CPUWalltimeNow() {
  static WallTimeImp& imp = WallTimeImp::GetWallTimeImp();
  return imp.Now();
}

WallTime ChronoWalltimeNow() {
  typedef ChooseClockType::type Clock;
  typedef std::chrono::duration<WallTime, std::chrono::seconds::period>
          FPSeconds;
  static_assert(std::chrono::treat_as_floating_point<WallTime>::value,
                "This type must be treated as a floating point type.");
  auto now = Clock::now().time_since_epoch();
  return std::chrono::duration_cast<FPSeconds>(now).count();
}

bool UseCpuCycleClock() {
    bool useWallTime = !CpuScalingEnabled();
    if (useWallTime) {
        VLOG(1) << "Using the CPU cycle clock to provide walltime::Now().\n";
    } else {
        VLOG(1) << "Using std::chrono to provide walltime::Now().\n";
    }
    return useWallTime;
}


} // end anonymous namespace

// WallTimeImp doesn't work when CPU Scaling is enabled. If CPU Scaling is
// enabled at the start of the program then std::chrono::system_clock is used
// instead.
WallTime Now()
{
  static bool useCPUClock = UseCpuCycleClock();
  if (useCPUClock) {
    return CPUWalltimeNow();
  } else {
    return ChronoWalltimeNow();
  }
}

}  // end namespace walltime


namespace {

std::string DateTimeString(bool local) {
  typedef std::chrono::system_clock Clock;
  std::time_t now = Clock::to_time_t(Clock::now());
  char storage[128];
  std::size_t written;

  if (local) {
#if defined(BENCHMARK_OS_WINDOWS)
    written = std::strftime(storage, sizeof(storage), "%x %X", ::localtime(&now));
#else
    std::tm timeinfo;
    std::memset(&timeinfo, 0, sizeof(std::tm));
    ::localtime_r(&now, &timeinfo);
    written = std::strftime(storage, sizeof(storage), "%F %T", &timeinfo);
#endif
  } else {
#if defined(BENCHMARK_OS_WINDOWS)
    written = std::strftime(storage, sizeof(storage), "%x %X", ::gmtime(&now));
#else
    std::tm timeinfo;
    std::memset(&timeinfo, 0, sizeof(std::tm));
    ::gmtime_r(&now, &timeinfo);
    written = std::strftime(storage, sizeof(storage), "%F %T", &timeinfo);
#endif
  }
  CHECK(written < arraysize(storage));
  ((void)written); // prevent unused variable in optimized mode.
  return std::string(storage);
}

} // end namespace

std::string LocalDateTimeString() {
  return DateTimeString(true);
}

}  // end namespace benchmark
