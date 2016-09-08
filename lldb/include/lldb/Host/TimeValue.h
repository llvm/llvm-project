//===-- TimeValue.h ---------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_TimeValue_h_
#define liblldb_TimeValue_h_

// C Includes
#include <stdint.h>
#ifndef _MSC_VER
#include <sys/time.h>

#endif

// C++ Includes
// Other libraries and framework includes
// Project includes
#include "lldb/lldb-private.h"

namespace lldb_private {

class TimeValue {
public:
  static const uint64_t MicroSecPerSec = 1000000UL;
  static const uint64_t NanoSecPerSec = 1000000000UL;
  static const uint64_t NanoSecPerMicroSec = 1000U;
  static const uint64_t NanoSecPerMilliSec = 1000000UL;

  //------------------------------------------------------------------
  // Constructors and Destructors
  //------------------------------------------------------------------
  TimeValue();
  TimeValue(const TimeValue &rhs);
  TimeValue(const struct timespec &ts);
  explicit TimeValue(uint32_t seconds, uint32_t nanos = 0);
  ~TimeValue();

  //------------------------------------------------------------------
  // Operators
  //------------------------------------------------------------------
  const TimeValue &operator=(const TimeValue &rhs);

  void Clear();

  uint64_t GetAsNanoSecondsSinceJan1_1970() const;

  uint64_t GetAsMicroSecondsSinceJan1_1970() const;

  uint64_t GetAsSecondsSinceJan1_1970() const;

  struct timespec GetAsTimeSpec() const;

  bool IsValid() const;

  void OffsetWithSeconds(uint64_t sec);

  void OffsetWithMicroSeconds(uint64_t usec);

  void OffsetWithNanoSeconds(uint64_t nsec);

  static TimeValue Now();

  void Dump(Stream *s, uint32_t width = 0) const;

  /// Returns only the seconds component of the TimeValue. The nanoseconds
  /// portion is ignored. No rounding is performed.
  /// @brief Retrieve the seconds component
  uint32_t seconds() const { return m_nano_seconds / NanoSecPerSec; }

  /// Returns only the nanoseconds component of the TimeValue. The seconds
  /// portion is ignored.
  /// @brief Retrieve the nanoseconds component.
  uint32_t nanoseconds() const { return m_nano_seconds % NanoSecPerSec; }

  /// Returns only the fractional portion of the TimeValue rounded down to the
  /// nearest microsecond (divide by one thousand).
  /// @brief Retrieve the fractional part as microseconds;
  uint32_t microseconds() const {
    return (m_nano_seconds % NanoSecPerSec) / NanoSecPerMicroSec;
  }

  /// Returns only the fractional portion of the TimeValue rounded down to the
  /// nearest millisecond (divide by one million).
  /// @brief Retrieve the milliseconds component;
  uint32_t milliseconds() const { return m_nano_seconds / NanoSecPerMilliSec; }

protected:
  //------------------------------------------------------------------
  // Classes that inherit from TimeValue can see and modify these
  //------------------------------------------------------------------
  uint64_t m_nano_seconds;
};

bool operator==(const TimeValue &lhs, const TimeValue &rhs);
bool operator!=(const TimeValue &lhs, const TimeValue &rhs);
bool operator<(const TimeValue &lhs, const TimeValue &rhs);
bool operator<=(const TimeValue &lhs, const TimeValue &rhs);
bool operator>(const TimeValue &lhs, const TimeValue &rhs);
bool operator>=(const TimeValue &lhs, const TimeValue &rhs);

uint64_t operator-(const TimeValue &lhs, const TimeValue &rhs);

} // namespace lldb_private

#endif // liblldb_TimeValue_h_
