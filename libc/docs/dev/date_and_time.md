(date_and_time)=

# Date and Time

LLVM-libc implements the C and POSIX date and time functions (`gmtime`,
`mktime`, etc.) using high-performance algorithms and a 64-bit `time_t`.
This page documents the design decisions and limitations of the implementation.

## Calendar Model

LLVM-libc uses the [proleptic Gregorian calendar](https://en.wikipedia.org/wiki/Proleptic_Gregorian_calendar): the modern
Gregorian leap-year rules (divisible by 4, except centuries, except
quad-centuries) are extended to all dates, including those before the
calendar's adoption on October 15, 1582.

This is the model required by the C and POSIX standards.  It means that dates
before 1582 do not correspond to historical Julian calendar dates.  For
example, `gmtime` will report February 29 for the year 400 CE, even though
that date was reckoned differently under the Julian calendar in use at the
time.

## Leap Seconds

POSIX §4.16 defines each day as exactly 86 400 seconds.  LLVM-libc follows
this convention: leap seconds are not represented and `time_t` values map
to UTC times that ignore leap-second insertions.

## 64-bit `time_t`

LLVM-libc requires a 64-bit `time_t`.  A 32-bit `time_t` is not
supported.  This avoids the [Year 2038 problem](https://en.wikipedia.org/wiki/Year_2038_problem) and provides a valid
range of approximately ±2 billion years from the Unix epoch (January 1, 1970).

## Algorithms

The date conversion functions use Ben Joffe's "Very Fast 64-bit" date
algorithm, which replaces the traditional loop-based year/month/day extraction
with multiply-shift division (four multiplications, zero hardware divisions).
The inverse (`mktime`) path uses the companion inverse algorithm from the
same article series.  Leap-year testing uses the Drepper–Neri–Schneider
algorithm (modulo 25 instead of modulo 100).

These algorithms are described in:

* [Very Fast 64-bit Date Algorithm](https://www.benjoffe.com/fast-date-64)
* [Inverse Date Algorithm](https://www.benjoffe.com/fast-date#inverse)
* [Fast Leap-Year Check](https://www.benjoffe.com/fast-leap-year#drepper-neri-schneider)

128-bit intermediate arithmetic is used for the multiply-shift steps.  On
64-bit targets this maps to the native `__uint128_t`; on 32-bit targets
LLVM-libc's software `UInt<128>` implementation is used, preserving
correctness at a modest performance cost.
