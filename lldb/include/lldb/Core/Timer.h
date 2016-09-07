//===-- Timer.h -------------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_Timer_h_
#define liblldb_Timer_h_

// C Includes
#include <stdarg.h>
#include <stdio.h>

// C++ Includes
#include <atomic>
#include <mutex>

// Other libraries and framework includes
// Project includes
#include "lldb/Host/TimeValue.h"
#include "lldb/lldb-private.h"

namespace lldb_private {

//----------------------------------------------------------------------
/// @class Timer Timer.h "lldb/Core/Timer.h"
/// @brief A timer class that simplifies common timing metrics.
///
/// A scoped timer class that allows a variety of pthread mutex
/// objects to have a mutex locked when a Timer::Locker
/// object is created, and unlocked when it goes out of scope or
/// when the Timer::Locker::Reset(pthread_mutex_t *)
/// is called. This provides an exception safe way to lock a mutex
/// in a scope.
//----------------------------------------------------------------------

class Timer {
public:
  //--------------------------------------------------------------
  /// Default constructor.
  //--------------------------------------------------------------
  Timer(const char *category, const char *format, ...)
      __attribute__((format(printf, 3, 4)));

  //--------------------------------------------------------------
  /// Destructor
  //--------------------------------------------------------------
  ~Timer();

  void Dump();

  static void SetDisplayDepth(uint32_t depth);

  static void SetQuiet(bool value);

  static void DumpCategoryTimes(Stream *s);

  static void ResetCategoryTimes();

protected:
  void ChildStarted(const TimeValue &time);

  void ChildStopped(const TimeValue &time);

  uint64_t GetTotalElapsedNanoSeconds();

  uint64_t GetTimerElapsedNanoSeconds();

  const char *m_category;
  TimeValue m_total_start;
  TimeValue m_timer_start;
  uint64_t m_total_ticks; // Total running time for this timer including when
                          // other timers below this are running
  uint64_t m_timer_ticks; // Ticks for this timer that do not include when other
                          // timers below this one are running

  static std::atomic<bool> g_quiet;
  static std::atomic<unsigned> g_display_depth;

private:
  Timer();
  DISALLOW_COPY_AND_ASSIGN(Timer);
};

class IntervalTimer {
public:
  IntervalTimer() : m_start(TimeValue::Now()) {}

  ~IntervalTimer() = default;

  uint64_t GetElapsedNanoSeconds() const { return TimeValue::Now() - m_start; }

  void Reset() { m_start = TimeValue::Now(); }

  int PrintfElapsed(const char *format, ...)
      __attribute__((format(printf, 2, 3))) {
    TimeValue now(TimeValue::Now());
    const uint64_t elapsed_nsec = now - m_start;
    const char *unit = nullptr;
    float elapsed_value;
    if (elapsed_nsec < 1000) {
      unit = "ns";
      elapsed_value = (float)elapsed_nsec;
    } else if (elapsed_nsec < 1000000) {
      unit = "us";
      elapsed_value = (float)elapsed_nsec / 1000.0f;
    } else if (elapsed_nsec < 1000000000) {
      unit = "ms";
      elapsed_value = (float)elapsed_nsec / 1000000.0f;
    } else {
      unit = "sec";
      elapsed_value = (float)elapsed_nsec / 1000000000.0f;
    }
    int result = printf("%3.2f %s: ", elapsed_value, unit);
    va_list args;
    va_start(args, format);
    result += vprintf(format, args);
    va_end(args);
    return result;
  }

protected:
  TimeValue m_start;
};

} // namespace lldb_private

#endif // liblldb_Timer_h_
