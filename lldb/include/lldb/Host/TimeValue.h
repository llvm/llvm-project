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
#include <sys/time.h>

// BEGIN: MinGW work around
#if !defined(_STRUCT_TIMESPEC) && !defined(HAVE_STRUCT_TIMESPEC)
#include <pthread.h>
#endif
// END: MinGW work around

// C++ Includes
// Other libraries and framework includes
// Project includes
#include "lldb/lldb-private.h"

namespace lldb_private {

class TimeValue
{
public:
    static const uint64_t MicroSecPerSec = 1000000UL;
    static const uint64_t NanoSecPerSec = 1000000000UL;
    static const uint64_t NanoSecPerMicroSec = 1000U;

    //------------------------------------------------------------------
    // Constructors and Destructors
    //------------------------------------------------------------------
    TimeValue();
    TimeValue(const TimeValue& rhs);
    TimeValue(const struct timespec& ts);
    TimeValue(const struct timeval& tv);
    ~TimeValue();

    //------------------------------------------------------------------
    // Operators
    //------------------------------------------------------------------
    const TimeValue&
    operator=(const TimeValue& rhs);

    void
    Clear ();

    uint64_t
    GetAsNanoSecondsSinceJan1_1970() const;

    uint64_t
    GetAsMicroSecondsSinceJan1_1970() const;

    uint64_t
    GetAsSecondsSinceJan1_1970() const;

    struct timespec
    GetAsTimeSpec () const;

    struct timeval
    GetAsTimeVal () const;

    bool
    IsValid () const;

    void
    OffsetWithSeconds (uint64_t sec);

    void
    OffsetWithMicroSeconds (uint64_t usec);

    void
    OffsetWithNanoSeconds (uint64_t nsec);

    static TimeValue
    Now();
    
    void
    Dump (Stream *s, uint32_t width = 0) const;

protected:
    //------------------------------------------------------------------
    // Classes that inherit from TimeValue can see and modify these
    //------------------------------------------------------------------
    uint64_t m_nano_seconds;
};

bool operator == (const TimeValue &lhs, const TimeValue &rhs);
bool operator != (const TimeValue &lhs, const TimeValue &rhs);
bool operator <  (const TimeValue &lhs, const TimeValue &rhs);
bool operator <= (const TimeValue &lhs, const TimeValue &rhs);
bool operator >  (const TimeValue &lhs, const TimeValue &rhs);
bool operator >= (const TimeValue &lhs, const TimeValue &rhs);

uint64_t operator -(const TimeValue &lhs, const TimeValue &rhs);

} // namespace lldb_private


#endif  // liblldb_TimeValue_h_
