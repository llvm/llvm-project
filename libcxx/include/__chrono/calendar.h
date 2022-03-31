// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___CHRONO_CALENDAR_H
#define _LIBCPP___CHRONO_CALENDAR_H

#include <__chrono/duration.h>
#include <__chrono/system_clock.h>
#include <__chrono/time_point.h>
#include <__config>
#include <limits>
#include <ratio>
#include <type_traits>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

_LIBCPP_PUSH_MACROS
#include <__undef_macros>

#if _LIBCPP_STD_VER > 17

_LIBCPP_BEGIN_NAMESPACE_STD

namespace chrono
{

struct local_t {};
template<class Duration>
using local_time  = time_point<local_t, Duration>;
using local_seconds = local_time<seconds>;
using local_days    = local_time<days>;

struct last_spec { _LIBCPP_HIDE_FROM_ABI explicit last_spec() = default; };

class day {
private:
    unsigned char __d;
public:
    _LIBCPP_HIDE_FROM_ABI day() = default;
    _LIBCPP_HIDE_FROM_ABI explicit inline constexpr day(unsigned __val) noexcept : __d(static_cast<unsigned char>(__val)) {}
    _LIBCPP_HIDE_FROM_ABI inline constexpr day& operator++()    noexcept { ++__d; return *this; }
    _LIBCPP_HIDE_FROM_ABI inline constexpr day  operator++(int) noexcept { day __tmp = *this; ++(*this); return __tmp; }
    _LIBCPP_HIDE_FROM_ABI inline constexpr day& operator--()    noexcept { --__d; return *this; }
    _LIBCPP_HIDE_FROM_ABI inline constexpr day  operator--(int) noexcept { day __tmp = *this; --(*this); return __tmp; }
    _LIBCPP_HIDE_FROM_ABI        constexpr day& operator+=(const days& __dd) noexcept;
    _LIBCPP_HIDE_FROM_ABI        constexpr day& operator-=(const days& __dd) noexcept;
    _LIBCPP_HIDE_FROM_ABI explicit inline constexpr operator unsigned() const noexcept { return __d; }
    _LIBCPP_HIDE_FROM_ABI inline constexpr bool ok() const noexcept { return __d >= 1 && __d <= 31; }
  };


_LIBCPP_HIDE_FROM_ABI inline constexpr
bool operator==(const day& __lhs, const day& __rhs) noexcept
{ return static_cast<unsigned>(__lhs) == static_cast<unsigned>(__rhs); }

_LIBCPP_HIDE_FROM_ABI inline constexpr
bool operator!=(const day& __lhs, const day& __rhs) noexcept
{ return !(__lhs == __rhs); }

_LIBCPP_HIDE_FROM_ABI inline constexpr
bool operator< (const day& __lhs, const day& __rhs) noexcept
{ return static_cast<unsigned>(__lhs) <  static_cast<unsigned>(__rhs); }

_LIBCPP_HIDE_FROM_ABI inline constexpr
bool operator> (const day& __lhs, const day& __rhs) noexcept
{ return __rhs < __lhs; }

_LIBCPP_HIDE_FROM_ABI inline constexpr
bool operator<=(const day& __lhs, const day& __rhs) noexcept
{ return !(__rhs < __lhs);}

_LIBCPP_HIDE_FROM_ABI inline constexpr
bool operator>=(const day& __lhs, const day& __rhs) noexcept
{ return !(__lhs < __rhs); }

_LIBCPP_HIDE_FROM_ABI inline constexpr
day operator+ (const day& __lhs, const days& __rhs) noexcept
{ return day(static_cast<unsigned>(__lhs) + __rhs.count()); }

_LIBCPP_HIDE_FROM_ABI inline constexpr
day operator+ (const days& __lhs, const day& __rhs) noexcept
{ return __rhs + __lhs; }

_LIBCPP_HIDE_FROM_ABI inline constexpr
day operator- (const day& __lhs, const days& __rhs) noexcept
{ return __lhs + -__rhs; }

_LIBCPP_HIDE_FROM_ABI inline constexpr
days operator-(const day& __lhs, const day& __rhs) noexcept
{ return days(static_cast<int>(static_cast<unsigned>(__lhs)) -
              static_cast<int>(static_cast<unsigned>(__rhs))); }

_LIBCPP_HIDE_FROM_ABI inline constexpr
day& day::operator+=(const days& __dd) noexcept
{ *this = *this + __dd; return *this; }

_LIBCPP_HIDE_FROM_ABI inline constexpr
day& day::operator-=(const days& __dd) noexcept
{ *this = *this - __dd; return *this; }


class month {
private:
    unsigned char __m;
public:
    _LIBCPP_HIDE_FROM_ABI month() = default;
    _LIBCPP_HIDE_FROM_ABI explicit inline constexpr month(unsigned __val) noexcept : __m(static_cast<unsigned char>(__val)) {}
    _LIBCPP_HIDE_FROM_ABI inline constexpr month& operator++()    noexcept { ++__m; return *this; }
    _LIBCPP_HIDE_FROM_ABI inline constexpr month  operator++(int) noexcept { month __tmp = *this; ++(*this); return __tmp; }
    _LIBCPP_HIDE_FROM_ABI inline constexpr month& operator--()    noexcept { --__m; return *this; }
    _LIBCPP_HIDE_FROM_ABI inline constexpr month  operator--(int) noexcept { month __tmp = *this; --(*this); return __tmp; }
    _LIBCPP_HIDE_FROM_ABI        constexpr month& operator+=(const months& __m1) noexcept;
    _LIBCPP_HIDE_FROM_ABI        constexpr month& operator-=(const months& __m1) noexcept;
    _LIBCPP_HIDE_FROM_ABI explicit inline constexpr operator unsigned() const noexcept { return __m; }
    _LIBCPP_HIDE_FROM_ABI inline constexpr bool ok() const noexcept { return __m >= 1 && __m <= 12; }
};


_LIBCPP_HIDE_FROM_ABI inline constexpr
bool operator==(const month& __lhs, const month& __rhs) noexcept
{ return static_cast<unsigned>(__lhs) == static_cast<unsigned>(__rhs); }

_LIBCPP_HIDE_FROM_ABI inline constexpr
bool operator!=(const month& __lhs, const month& __rhs) noexcept
{ return !(__lhs == __rhs); }

_LIBCPP_HIDE_FROM_ABI inline constexpr
bool operator< (const month& __lhs, const month& __rhs) noexcept
{ return static_cast<unsigned>(__lhs)  < static_cast<unsigned>(__rhs); }

_LIBCPP_HIDE_FROM_ABI inline constexpr
bool operator> (const month& __lhs, const month& __rhs) noexcept
{ return __rhs < __lhs; }

_LIBCPP_HIDE_FROM_ABI inline constexpr
bool operator<=(const month& __lhs, const month& __rhs) noexcept
{ return !(__rhs < __lhs); }

_LIBCPP_HIDE_FROM_ABI inline constexpr
bool operator>=(const month& __lhs, const month& __rhs) noexcept
{ return !(__lhs < __rhs); }

_LIBCPP_HIDE_FROM_ABI inline constexpr
month operator+ (const month& __lhs, const months& __rhs) noexcept
{
    auto const __mu = static_cast<long long>(static_cast<unsigned>(__lhs)) + (__rhs.count() - 1);
    auto const __yr = (__mu >= 0 ? __mu : __mu - 11) / 12;
    return month{static_cast<unsigned>(__mu - __yr * 12 + 1)};
}

_LIBCPP_HIDE_FROM_ABI inline constexpr
month operator+ (const months& __lhs, const month& __rhs) noexcept
{ return __rhs + __lhs; }

_LIBCPP_HIDE_FROM_ABI inline constexpr
month operator- (const month& __lhs, const months& __rhs) noexcept
{ return __lhs + -__rhs; }

_LIBCPP_HIDE_FROM_ABI inline constexpr
months operator-(const month& __lhs, const month& __rhs) noexcept
{
    auto const __dm = static_cast<unsigned>(__lhs) - static_cast<unsigned>(__rhs);
    return months(__dm <= 11 ? __dm : __dm + 12);
}

_LIBCPP_HIDE_FROM_ABI inline constexpr
month& month::operator+=(const months& __dm) noexcept
{ *this = *this + __dm; return *this; }

_LIBCPP_HIDE_FROM_ABI inline constexpr
month& month::operator-=(const months& __dm) noexcept
{ *this = *this - __dm; return *this; }


class year {
private:
    short __y;
public:
    _LIBCPP_HIDE_FROM_ABI year() = default;
    _LIBCPP_HIDE_FROM_ABI explicit inline constexpr year(int __val) noexcept : __y(static_cast<short>(__val)) {}

    _LIBCPP_HIDE_FROM_ABI inline constexpr year& operator++()    noexcept { ++__y; return *this; }
    _LIBCPP_HIDE_FROM_ABI inline constexpr year  operator++(int) noexcept { year __tmp = *this; ++(*this); return __tmp; }
    _LIBCPP_HIDE_FROM_ABI inline constexpr year& operator--()    noexcept { --__y; return *this; }
    _LIBCPP_HIDE_FROM_ABI inline constexpr year  operator--(int) noexcept { year __tmp = *this; --(*this); return __tmp; }
    _LIBCPP_HIDE_FROM_ABI        constexpr year& operator+=(const years& __dy) noexcept;
    _LIBCPP_HIDE_FROM_ABI        constexpr year& operator-=(const years& __dy) noexcept;
    _LIBCPP_HIDE_FROM_ABI inline constexpr year operator+() const noexcept { return *this; }
    _LIBCPP_HIDE_FROM_ABI inline constexpr year operator-() const noexcept { return year{-__y}; }

    _LIBCPP_HIDE_FROM_ABI inline constexpr bool is_leap() const noexcept { return __y % 4 == 0 && (__y % 100 != 0 || __y % 400 == 0); }
    _LIBCPP_HIDE_FROM_ABI explicit inline constexpr operator int() const noexcept { return __y; }
    _LIBCPP_HIDE_FROM_ABI        constexpr bool ok() const noexcept;
    _LIBCPP_HIDE_FROM_ABI static inline constexpr year min() noexcept { return year{-32767}; }
    _LIBCPP_HIDE_FROM_ABI static inline constexpr year max() noexcept { return year{ 32767}; }
};


_LIBCPP_HIDE_FROM_ABI inline constexpr
bool operator==(const year& __lhs, const year& __rhs) noexcept
{ return static_cast<int>(__lhs) == static_cast<int>(__rhs); }

_LIBCPP_HIDE_FROM_ABI inline constexpr
bool operator!=(const year& __lhs, const year& __rhs) noexcept
{ return !(__lhs == __rhs); }

_LIBCPP_HIDE_FROM_ABI inline constexpr
bool operator< (const year& __lhs, const year& __rhs) noexcept
{ return static_cast<int>(__lhs)  < static_cast<int>(__rhs); }

_LIBCPP_HIDE_FROM_ABI inline constexpr
bool operator> (const year& __lhs, const year& __rhs) noexcept
{ return __rhs < __lhs; }

_LIBCPP_HIDE_FROM_ABI inline constexpr
bool operator<=(const year& __lhs, const year& __rhs) noexcept
{ return !(__rhs < __lhs); }

_LIBCPP_HIDE_FROM_ABI inline constexpr
bool operator>=(const year& __lhs, const year& __rhs) noexcept
{ return !(__lhs < __rhs); }

_LIBCPP_HIDE_FROM_ABI inline constexpr
year operator+ (const year& __lhs, const years& __rhs) noexcept
{ return year(static_cast<int>(__lhs) + __rhs.count()); }

_LIBCPP_HIDE_FROM_ABI inline constexpr
year operator+ (const years& __lhs, const year& __rhs) noexcept
{ return __rhs + __lhs; }

_LIBCPP_HIDE_FROM_ABI inline constexpr
year operator- (const year& __lhs, const years& __rhs) noexcept
{ return __lhs + -__rhs; }

_LIBCPP_HIDE_FROM_ABI inline constexpr
years operator-(const year& __lhs, const year& __rhs) noexcept
{ return years{static_cast<int>(__lhs) - static_cast<int>(__rhs)}; }


_LIBCPP_HIDE_FROM_ABI inline constexpr
year& year::operator+=(const years& __dy) noexcept
{ *this = *this + __dy; return *this; }

_LIBCPP_HIDE_FROM_ABI inline constexpr
year& year::operator-=(const years& __dy) noexcept
{ *this = *this - __dy; return *this; }

_LIBCPP_HIDE_FROM_ABI inline constexpr
bool year::ok() const noexcept
{ return static_cast<int>(min()) <= __y && __y <= static_cast<int>(max()); }

class weekday_indexed;
class weekday_last;

class weekday {
private:
    unsigned char __wd;
    _LIBCPP_HIDE_FROM_ABI static constexpr unsigned char __weekday_from_days(int __days) noexcept;
public:
  _LIBCPP_HIDE_FROM_ABI weekday() = default;
  _LIBCPP_HIDE_FROM_ABI inline explicit constexpr weekday(unsigned __val) noexcept : __wd(static_cast<unsigned char>(__val == 7 ? 0 : __val)) {}
  _LIBCPP_HIDE_FROM_ABI inline constexpr          weekday(const sys_days& __sysd) noexcept
          : __wd(__weekday_from_days(__sysd.time_since_epoch().count())) {}
  _LIBCPP_HIDE_FROM_ABI inline explicit constexpr weekday(const local_days& __locd) noexcept
          : __wd(__weekday_from_days(__locd.time_since_epoch().count())) {}

  _LIBCPP_HIDE_FROM_ABI inline constexpr weekday& operator++()    noexcept { __wd = (__wd == 6 ? 0 : __wd + 1); return *this; }
  _LIBCPP_HIDE_FROM_ABI inline constexpr weekday  operator++(int) noexcept { weekday __tmp = *this; ++(*this); return __tmp; }
  _LIBCPP_HIDE_FROM_ABI inline constexpr weekday& operator--()    noexcept { __wd = (__wd == 0 ? 6 : __wd - 1); return *this; }
  _LIBCPP_HIDE_FROM_ABI inline constexpr weekday  operator--(int) noexcept { weekday __tmp = *this; --(*this); return __tmp; }
  _LIBCPP_HIDE_FROM_ABI        constexpr weekday& operator+=(const days& __dd) noexcept;
  _LIBCPP_HIDE_FROM_ABI        constexpr weekday& operator-=(const days& __dd) noexcept;
  _LIBCPP_HIDE_FROM_ABI inline constexpr unsigned c_encoding()   const noexcept { return __wd; }
  _LIBCPP_HIDE_FROM_ABI inline constexpr unsigned iso_encoding() const noexcept { return __wd == 0u ? 7 : __wd; }
  _LIBCPP_HIDE_FROM_ABI inline constexpr bool ok() const noexcept { return __wd <= 6; }
  _LIBCPP_HIDE_FROM_ABI        constexpr weekday_indexed operator[](unsigned __index) const noexcept;
  _LIBCPP_HIDE_FROM_ABI        constexpr weekday_last    operator[](last_spec) const noexcept;
};


// https://howardhinnant.github.io/date_algorithms.html#weekday_from_days
_LIBCPP_HIDE_FROM_ABI inline constexpr
unsigned char weekday::__weekday_from_days(int __days) noexcept
{
    return static_cast<unsigned char>(
              static_cast<unsigned>(__days >= -4 ? (__days+4) % 7 : (__days+5) % 7 + 6)
           );
}

_LIBCPP_HIDE_FROM_ABI inline constexpr
bool operator==(const weekday& __lhs, const weekday& __rhs) noexcept
{ return __lhs.c_encoding() == __rhs.c_encoding(); }

_LIBCPP_HIDE_FROM_ABI inline constexpr
bool operator!=(const weekday& __lhs, const weekday& __rhs) noexcept
{ return !(__lhs == __rhs); }

_LIBCPP_HIDE_FROM_ABI inline constexpr
bool operator< (const weekday& __lhs, const weekday& __rhs) noexcept
{ return __lhs.c_encoding() < __rhs.c_encoding(); }

_LIBCPP_HIDE_FROM_ABI inline constexpr
bool operator> (const weekday& __lhs, const weekday& __rhs) noexcept
{ return __rhs < __lhs; }

_LIBCPP_HIDE_FROM_ABI inline constexpr
bool operator<=(const weekday& __lhs, const weekday& __rhs) noexcept
{ return !(__rhs < __lhs);}

_LIBCPP_HIDE_FROM_ABI inline constexpr
bool operator>=(const weekday& __lhs, const weekday& __rhs) noexcept
{ return !(__lhs < __rhs); }

_LIBCPP_HIDE_FROM_ABI constexpr
weekday operator+(const weekday& __lhs, const days& __rhs) noexcept
{
    auto const __mu = static_cast<long long>(__lhs.c_encoding()) + __rhs.count();
    auto const __yr = (__mu >= 0 ? __mu : __mu - 6) / 7;
    return weekday{static_cast<unsigned>(__mu - __yr * 7)};
}

_LIBCPP_HIDE_FROM_ABI constexpr
weekday operator+(const days& __lhs, const weekday& __rhs) noexcept
{ return __rhs + __lhs; }

_LIBCPP_HIDE_FROM_ABI constexpr
weekday operator-(const weekday& __lhs, const days& __rhs) noexcept
{ return __lhs + -__rhs; }

_LIBCPP_HIDE_FROM_ABI constexpr
days operator-(const weekday& __lhs, const weekday& __rhs) noexcept
{
    const int __wdu = __lhs.c_encoding() - __rhs.c_encoding();
    const int __wk = (__wdu >= 0 ? __wdu : __wdu-6) / 7;
    return days{__wdu - __wk * 7};
}

_LIBCPP_HIDE_FROM_ABI inline constexpr
weekday& weekday::operator+=(const days& __dd) noexcept
{ *this = *this + __dd; return *this; }

_LIBCPP_HIDE_FROM_ABI inline constexpr
weekday& weekday::operator-=(const days& __dd) noexcept
{ *this = *this - __dd; return *this; }


class weekday_indexed {
private:
    chrono::weekday __wd;
    unsigned char   __idx;
public:
    _LIBCPP_HIDE_FROM_ABI weekday_indexed() = default;
    _LIBCPP_HIDE_FROM_ABI inline constexpr weekday_indexed(const chrono::weekday& __wdval, unsigned __idxval) noexcept
        : __wd{__wdval}, __idx(__idxval) {}
    _LIBCPP_HIDE_FROM_ABI inline constexpr chrono::weekday weekday() const noexcept { return __wd; }
    _LIBCPP_HIDE_FROM_ABI inline constexpr unsigned                 index() const noexcept { return __idx; }
    _LIBCPP_HIDE_FROM_ABI inline constexpr bool ok() const noexcept { return __wd.ok() && __idx >= 1 && __idx <= 5; }
};

_LIBCPP_HIDE_FROM_ABI inline constexpr
bool operator==(const weekday_indexed& __lhs, const weekday_indexed& __rhs) noexcept
{ return __lhs.weekday() == __rhs.weekday() && __lhs.index() == __rhs.index(); }

_LIBCPP_HIDE_FROM_ABI inline constexpr
bool operator!=(const weekday_indexed& __lhs, const weekday_indexed& __rhs) noexcept
{ return !(__lhs == __rhs); }


class weekday_last {
private:
    chrono::weekday __wd;
public:
    _LIBCPP_HIDE_FROM_ABI explicit constexpr weekday_last(const chrono::weekday& __val) noexcept
        : __wd{__val} {}
    _LIBCPP_HIDE_FROM_ABI constexpr chrono::weekday weekday() const noexcept { return __wd; }
    _LIBCPP_HIDE_FROM_ABI constexpr bool ok() const noexcept { return __wd.ok(); }
};

_LIBCPP_HIDE_FROM_ABI inline constexpr
bool operator==(const weekday_last& __lhs, const weekday_last& __rhs) noexcept
{ return __lhs.weekday() == __rhs.weekday(); }

_LIBCPP_HIDE_FROM_ABI inline constexpr
bool operator!=(const weekday_last& __lhs, const weekday_last& __rhs) noexcept
{ return !(__lhs == __rhs); }

_LIBCPP_HIDE_FROM_ABI inline constexpr
weekday_indexed weekday::operator[](unsigned __index) const noexcept { return weekday_indexed{*this, __index}; }

_LIBCPP_HIDE_FROM_ABI inline constexpr
weekday_last weekday::operator[](last_spec) const noexcept { return weekday_last{*this}; }


inline constexpr last_spec last{};
inline constexpr weekday   Sunday{0};
inline constexpr weekday   Monday{1};
inline constexpr weekday   Tuesday{2};
inline constexpr weekday   Wednesday{3};
inline constexpr weekday   Thursday{4};
inline constexpr weekday   Friday{5};
inline constexpr weekday   Saturday{6};

inline constexpr month January{1};
inline constexpr month February{2};
inline constexpr month March{3};
inline constexpr month April{4};
inline constexpr month May{5};
inline constexpr month June{6};
inline constexpr month July{7};
inline constexpr month August{8};
inline constexpr month September{9};
inline constexpr month October{10};
inline constexpr month November{11};
inline constexpr month December{12};


class month_day {
private:
   chrono::month __m;
   chrono::day   __d;
public:
    _LIBCPP_HIDE_FROM_ABI month_day() = default;
    _LIBCPP_HIDE_FROM_ABI constexpr month_day(const chrono::month& __mval, const chrono::day& __dval) noexcept
        : __m{__mval}, __d{__dval} {}
    _LIBCPP_HIDE_FROM_ABI inline constexpr chrono::month month() const noexcept { return __m; }
    _LIBCPP_HIDE_FROM_ABI inline constexpr chrono::day   day()   const noexcept { return __d; }
    _LIBCPP_HIDE_FROM_ABI constexpr bool ok() const noexcept;
};

_LIBCPP_HIDE_FROM_ABI inline constexpr
bool month_day::ok() const noexcept
{
    if (!__m.ok()) return false;
    const unsigned __dval = static_cast<unsigned>(__d);
    if (__dval < 1 || __dval > 31) return false;
    if (__dval <= 29) return true;
//  Now we've got either 30 or 31
    const unsigned __mval = static_cast<unsigned>(__m);
    if (__mval == 2) return false;
    if (__mval == 4 || __mval == 6 || __mval == 9 || __mval == 11)
        return __dval == 30;
    return true;
}

_LIBCPP_HIDE_FROM_ABI inline constexpr
bool operator==(const month_day& __lhs, const month_day& __rhs) noexcept
{ return __lhs.month() == __rhs.month() && __lhs.day() == __rhs.day(); }

_LIBCPP_HIDE_FROM_ABI inline constexpr
bool operator!=(const month_day& __lhs, const month_day& __rhs) noexcept
{ return !(__lhs == __rhs); }

_LIBCPP_HIDE_FROM_ABI inline constexpr
month_day operator/(const month& __lhs, const day& __rhs) noexcept
{ return month_day{__lhs, __rhs}; }

_LIBCPP_HIDE_FROM_ABI constexpr
month_day operator/(const day& __lhs, const month& __rhs) noexcept
{ return __rhs / __lhs; }

_LIBCPP_HIDE_FROM_ABI inline constexpr
month_day operator/(const month& __lhs, int __rhs) noexcept
{ return __lhs / day(__rhs); }

_LIBCPP_HIDE_FROM_ABI constexpr
month_day operator/(int __lhs, const day& __rhs) noexcept
{ return month(__lhs) / __rhs; }

_LIBCPP_HIDE_FROM_ABI constexpr
month_day operator/(const day& __lhs, int __rhs) noexcept
{ return month(__rhs) / __lhs; }


_LIBCPP_HIDE_FROM_ABI inline constexpr
bool operator< (const month_day& __lhs, const month_day& __rhs) noexcept
{ return __lhs.month() != __rhs.month() ? __lhs.month() < __rhs.month() : __lhs.day() < __rhs.day(); }

_LIBCPP_HIDE_FROM_ABI inline constexpr
bool operator> (const month_day& __lhs, const month_day& __rhs) noexcept
{ return __rhs < __lhs; }

_LIBCPP_HIDE_FROM_ABI inline constexpr
bool operator<=(const month_day& __lhs, const month_day& __rhs) noexcept
{ return !(__rhs < __lhs);}

_LIBCPP_HIDE_FROM_ABI inline constexpr
bool operator>=(const month_day& __lhs, const month_day& __rhs) noexcept
{ return !(__lhs < __rhs); }



class month_day_last {
private:
    chrono::month __m;
public:
    _LIBCPP_HIDE_FROM_ABI explicit constexpr month_day_last(const chrono::month& __val) noexcept
        : __m{__val} {}
    _LIBCPP_HIDE_FROM_ABI inline constexpr chrono::month month() const noexcept { return __m; }
    _LIBCPP_HIDE_FROM_ABI inline constexpr bool ok() const noexcept { return __m.ok(); }
};

_LIBCPP_HIDE_FROM_ABI inline constexpr
bool operator==(const month_day_last& __lhs, const month_day_last& __rhs) noexcept
{ return __lhs.month() == __rhs.month(); }

_LIBCPP_HIDE_FROM_ABI inline constexpr
bool operator!=(const month_day_last& __lhs, const month_day_last& __rhs) noexcept
{ return !(__lhs == __rhs); }

_LIBCPP_HIDE_FROM_ABI inline constexpr
bool operator< (const month_day_last& __lhs, const month_day_last& __rhs) noexcept
{ return __lhs.month() < __rhs.month(); }

_LIBCPP_HIDE_FROM_ABI inline constexpr
bool operator> (const month_day_last& __lhs, const month_day_last& __rhs) noexcept
{ return __rhs < __lhs; }

_LIBCPP_HIDE_FROM_ABI inline constexpr
bool operator<=(const month_day_last& __lhs, const month_day_last& __rhs) noexcept
{ return !(__rhs < __lhs);}

_LIBCPP_HIDE_FROM_ABI inline constexpr
bool operator>=(const month_day_last& __lhs, const month_day_last& __rhs) noexcept
{ return !(__lhs < __rhs); }

_LIBCPP_HIDE_FROM_ABI inline constexpr
month_day_last operator/(const month& __lhs, last_spec) noexcept
{ return month_day_last{__lhs}; }

_LIBCPP_HIDE_FROM_ABI inline constexpr
month_day_last operator/(last_spec, const month& __rhs) noexcept
{ return month_day_last{__rhs}; }

_LIBCPP_HIDE_FROM_ABI inline constexpr
month_day_last operator/(int __lhs, last_spec) noexcept
{ return month_day_last{month(__lhs)}; }

_LIBCPP_HIDE_FROM_ABI inline constexpr
month_day_last operator/(last_spec, int __rhs) noexcept
{ return month_day_last{month(__rhs)}; }


class month_weekday {
private:
    chrono::month __m;
    chrono::weekday_indexed __wdi;
public:
    _LIBCPP_HIDE_FROM_ABI constexpr month_weekday(const chrono::month& __mval, const chrono::weekday_indexed& __wdival) noexcept
        : __m{__mval}, __wdi{__wdival} {}
    _LIBCPP_HIDE_FROM_ABI inline constexpr chrono::month                     month() const noexcept { return __m; }
    _LIBCPP_HIDE_FROM_ABI inline constexpr chrono::weekday_indexed weekday_indexed() const noexcept { return __wdi; }
    _LIBCPP_HIDE_FROM_ABI inline constexpr bool                                 ok() const noexcept { return __m.ok() && __wdi.ok(); }
};

_LIBCPP_HIDE_FROM_ABI inline constexpr
bool operator==(const month_weekday& __lhs, const month_weekday& __rhs) noexcept
{ return __lhs.month() == __rhs.month() && __lhs.weekday_indexed() == __rhs.weekday_indexed(); }

_LIBCPP_HIDE_FROM_ABI inline constexpr
bool operator!=(const month_weekday& __lhs, const month_weekday& __rhs) noexcept
{ return !(__lhs == __rhs); }

_LIBCPP_HIDE_FROM_ABI inline constexpr
month_weekday operator/(const month& __lhs, const weekday_indexed& __rhs) noexcept
{ return month_weekday{__lhs, __rhs}; }

_LIBCPP_HIDE_FROM_ABI inline constexpr
month_weekday operator/(int __lhs, const weekday_indexed& __rhs) noexcept
{ return month_weekday{month(__lhs), __rhs}; }

_LIBCPP_HIDE_FROM_ABI inline constexpr
month_weekday operator/(const weekday_indexed& __lhs, const month& __rhs) noexcept
{ return month_weekday{__rhs, __lhs}; }

_LIBCPP_HIDE_FROM_ABI inline constexpr
month_weekday operator/(const weekday_indexed& __lhs, int __rhs) noexcept
{ return month_weekday{month(__rhs), __lhs}; }


class month_weekday_last {
    chrono::month        __m;
    chrono::weekday_last __wdl;
  public:
    _LIBCPP_HIDE_FROM_ABI constexpr month_weekday_last(const chrono::month& __mval, const chrono::weekday_last& __wdlval) noexcept
        : __m{__mval}, __wdl{__wdlval} {}
    _LIBCPP_HIDE_FROM_ABI inline constexpr chrono::month               month() const noexcept { return __m; }
    _LIBCPP_HIDE_FROM_ABI inline constexpr chrono::weekday_last weekday_last() const noexcept { return __wdl; }
    _LIBCPP_HIDE_FROM_ABI inline constexpr bool                           ok() const noexcept { return __m.ok() && __wdl.ok(); }
};

_LIBCPP_HIDE_FROM_ABI inline constexpr
bool operator==(const month_weekday_last& __lhs, const month_weekday_last& __rhs) noexcept
{ return __lhs.month() == __rhs.month() && __lhs.weekday_last() == __rhs.weekday_last(); }

_LIBCPP_HIDE_FROM_ABI inline constexpr
bool operator!=(const month_weekday_last& __lhs, const month_weekday_last& __rhs) noexcept
{ return !(__lhs == __rhs); }


_LIBCPP_HIDE_FROM_ABI inline constexpr
month_weekday_last operator/(const month& __lhs, const weekday_last& __rhs) noexcept
{ return month_weekday_last{__lhs, __rhs}; }

_LIBCPP_HIDE_FROM_ABI inline constexpr
month_weekday_last operator/(int __lhs, const weekday_last& __rhs) noexcept
{ return month_weekday_last{month(__lhs), __rhs}; }

_LIBCPP_HIDE_FROM_ABI inline constexpr
month_weekday_last operator/(const weekday_last& __lhs, const month& __rhs) noexcept
{ return month_weekday_last{__rhs, __lhs}; }

_LIBCPP_HIDE_FROM_ABI inline constexpr
month_weekday_last operator/(const weekday_last& __lhs, int __rhs) noexcept
{ return month_weekday_last{month(__rhs), __lhs}; }


class year_month {
    chrono::year  __y;
    chrono::month __m;
public:
    _LIBCPP_HIDE_FROM_ABI year_month() = default;
    _LIBCPP_HIDE_FROM_ABI constexpr year_month(const chrono::year& __yval, const chrono::month& __mval) noexcept
        : __y{__yval}, __m{__mval} {}
    _LIBCPP_HIDE_FROM_ABI inline constexpr chrono::year  year()  const noexcept { return __y; }
    _LIBCPP_HIDE_FROM_ABI inline constexpr chrono::month month() const noexcept { return __m; }
    _LIBCPP_HIDE_FROM_ABI inline constexpr year_month& operator+=(const months& __dm) noexcept { this->__m += __dm; return *this; }
    _LIBCPP_HIDE_FROM_ABI inline constexpr year_month& operator-=(const months& __dm) noexcept { this->__m -= __dm; return *this; }
    _LIBCPP_HIDE_FROM_ABI inline constexpr year_month& operator+=(const years& __dy)  noexcept { this->__y += __dy; return *this; }
    _LIBCPP_HIDE_FROM_ABI inline constexpr year_month& operator-=(const years& __dy)  noexcept { this->__y -= __dy; return *this; }
    _LIBCPP_HIDE_FROM_ABI inline constexpr bool ok() const noexcept { return __y.ok() && __m.ok(); }
};

_LIBCPP_HIDE_FROM_ABI inline constexpr
year_month operator/(const year& __y, const month& __m) noexcept { return year_month{__y, __m}; }

_LIBCPP_HIDE_FROM_ABI inline constexpr
year_month operator/(const year& __y, int __m) noexcept { return year_month{__y, month(__m)}; }

_LIBCPP_HIDE_FROM_ABI inline constexpr
bool operator==(const year_month& __lhs, const year_month& __rhs) noexcept
{ return __lhs.year() == __rhs.year() && __lhs.month() == __rhs.month(); }

_LIBCPP_HIDE_FROM_ABI inline constexpr
bool operator!=(const year_month& __lhs, const year_month& __rhs) noexcept
{ return !(__lhs == __rhs); }

_LIBCPP_HIDE_FROM_ABI inline constexpr
bool operator< (const year_month& __lhs, const year_month& __rhs) noexcept
{ return __lhs.year() != __rhs.year() ? __lhs.year() < __rhs.year() : __lhs.month() < __rhs.month(); }

_LIBCPP_HIDE_FROM_ABI inline constexpr
bool operator> (const year_month& __lhs, const year_month& __rhs) noexcept
{ return __rhs < __lhs; }

_LIBCPP_HIDE_FROM_ABI inline constexpr
bool operator<=(const year_month& __lhs, const year_month& __rhs) noexcept
{ return !(__rhs < __lhs);}

_LIBCPP_HIDE_FROM_ABI inline constexpr
bool operator>=(const year_month& __lhs, const year_month& __rhs) noexcept
{ return !(__lhs < __rhs); }

_LIBCPP_HIDE_FROM_ABI constexpr
year_month operator+(const year_month& __lhs, const months& __rhs) noexcept
{
    int __dmi = static_cast<int>(static_cast<unsigned>(__lhs.month())) - 1 + __rhs.count();
    const int __dy = (__dmi >= 0 ? __dmi : __dmi-11) / 12;
    __dmi = __dmi - __dy * 12 + 1;
    return (__lhs.year() + years(__dy)) / month(static_cast<unsigned>(__dmi));
}

_LIBCPP_HIDE_FROM_ABI constexpr
year_month operator+(const months& __lhs, const year_month& __rhs) noexcept
{ return __rhs + __lhs; }

_LIBCPP_HIDE_FROM_ABI constexpr
year_month operator+(const year_month& __lhs, const years& __rhs) noexcept
{ return (__lhs.year() + __rhs) / __lhs.month(); }

_LIBCPP_HIDE_FROM_ABI constexpr
year_month operator+(const years& __lhs, const year_month& __rhs) noexcept
{ return __rhs + __lhs; }

_LIBCPP_HIDE_FROM_ABI constexpr
months operator-(const year_month& __lhs, const year_month& __rhs) noexcept
{ return (__lhs.year() - __rhs.year()) + months(static_cast<unsigned>(__lhs.month()) - static_cast<unsigned>(__rhs.month())); }

_LIBCPP_HIDE_FROM_ABI constexpr
year_month operator-(const year_month& __lhs, const months& __rhs) noexcept
{ return __lhs + -__rhs; }

_LIBCPP_HIDE_FROM_ABI constexpr
year_month operator-(const year_month& __lhs, const years& __rhs) noexcept
{ return __lhs + -__rhs; }

class year_month_day_last;

class year_month_day {
private:
    chrono::year  __y;
    chrono::month __m;
    chrono::day   __d;
public:
     _LIBCPP_HIDE_FROM_ABI year_month_day() = default;
     _LIBCPP_HIDE_FROM_ABI inline constexpr year_month_day(
            const chrono::year& __yval, const chrono::month& __mval, const chrono::day& __dval) noexcept
            : __y{__yval}, __m{__mval}, __d{__dval} {}
     _LIBCPP_HIDE_FROM_ABI        constexpr year_month_day(const year_month_day_last& __ymdl) noexcept;
     _LIBCPP_HIDE_FROM_ABI inline constexpr year_month_day(const sys_days& __sysd) noexcept
            : year_month_day(__from_days(__sysd.time_since_epoch())) {}
     _LIBCPP_HIDE_FROM_ABI inline explicit constexpr year_month_day(const local_days& __locd) noexcept
            : year_month_day(__from_days(__locd.time_since_epoch())) {}

     _LIBCPP_HIDE_FROM_ABI        constexpr year_month_day& operator+=(const months& __dm) noexcept;
     _LIBCPP_HIDE_FROM_ABI        constexpr year_month_day& operator-=(const months& __dm) noexcept;
     _LIBCPP_HIDE_FROM_ABI        constexpr year_month_day& operator+=(const years& __dy)  noexcept;
     _LIBCPP_HIDE_FROM_ABI        constexpr year_month_day& operator-=(const years& __dy)  noexcept;

     _LIBCPP_HIDE_FROM_ABI inline constexpr chrono::year   year() const noexcept { return __y; }
     _LIBCPP_HIDE_FROM_ABI inline constexpr chrono::month month() const noexcept { return __m; }
     _LIBCPP_HIDE_FROM_ABI inline constexpr chrono::day     day() const noexcept { return __d; }
     _LIBCPP_HIDE_FROM_ABI inline constexpr operator   sys_days() const noexcept          { return   sys_days{__to_days()}; }
     _LIBCPP_HIDE_FROM_ABI inline explicit constexpr operator local_days() const noexcept { return local_days{__to_days()}; }

     _LIBCPP_HIDE_FROM_ABI        constexpr bool             ok() const noexcept;

     _LIBCPP_HIDE_FROM_ABI static constexpr year_month_day __from_days(days __d) noexcept;
     _LIBCPP_HIDE_FROM_ABI constexpr days __to_days() const noexcept;
};


// https://howardhinnant.github.io/date_algorithms.html#civil_from_days
_LIBCPP_HIDE_FROM_ABI inline constexpr
year_month_day year_month_day::__from_days(days __d) noexcept
{
    static_assert(numeric_limits<unsigned>::digits >= 18, "");
    static_assert(numeric_limits<int>::digits >= 20     , "");
    const int      __z = __d.count() + 719468;
    const int      __era = (__z >= 0 ? __z : __z - 146096) / 146097;
    const unsigned __doe = static_cast<unsigned>(__z - __era * 146097);              // [0, 146096]
    const unsigned __yoe = (__doe - __doe/1460 + __doe/36524 - __doe/146096) / 365;  // [0, 399]
    const int      __yr = static_cast<int>(__yoe) + __era * 400;
    const unsigned __doy = __doe - (365 * __yoe + __yoe/4 - __yoe/100);              // [0, 365]
    const unsigned __mp = (5 * __doy + 2)/153;                                       // [0, 11]
    const unsigned __dy = __doy - (153 * __mp + 2)/5 + 1;                            // [1, 31]
    const unsigned __mth = __mp + (__mp < 10 ? 3 : -9);                              // [1, 12]
    return year_month_day{chrono::year{__yr + (__mth <= 2)}, chrono::month{__mth}, chrono::day{__dy}};
}

// https://howardhinnant.github.io/date_algorithms.html#days_from_civil
_LIBCPP_HIDE_FROM_ABI inline constexpr
days year_month_day::__to_days() const noexcept
{
    static_assert(numeric_limits<unsigned>::digits >= 18, "");
    static_assert(numeric_limits<int>::digits >= 20     , "");

    const int      __yr  = static_cast<int>(__y) - (__m <= February);
    const unsigned __mth = static_cast<unsigned>(__m);
    const unsigned __dy  = static_cast<unsigned>(__d);

    const int      __era = (__yr >= 0 ? __yr : __yr - 399) / 400;
    const unsigned __yoe = static_cast<unsigned>(__yr - __era * 400);                // [0, 399]
    const unsigned __doy = (153 * (__mth + (__mth > 2 ? -3 : 9)) + 2) / 5 + __dy-1;  // [0, 365]
    const unsigned __doe = __yoe * 365 + __yoe/4 - __yoe/100 + __doy;                // [0, 146096]
    return days{__era * 146097 + static_cast<int>(__doe) - 719468};
}

_LIBCPP_HIDE_FROM_ABI inline constexpr
bool operator==(const year_month_day& __lhs, const year_month_day& __rhs) noexcept
{ return __lhs.year() == __rhs.year() && __lhs.month() == __rhs.month() && __lhs.day() == __rhs.day(); }

_LIBCPP_HIDE_FROM_ABI inline constexpr
bool operator!=(const year_month_day& __lhs, const year_month_day& __rhs) noexcept
{ return !(__lhs == __rhs); }

_LIBCPP_HIDE_FROM_ABI inline constexpr
bool operator< (const year_month_day& __lhs, const year_month_day& __rhs) noexcept
{
    if (__lhs.year() < __rhs.year()) return true;
    if (__lhs.year() > __rhs.year()) return false;
    if (__lhs.month() < __rhs.month()) return true;
    if (__lhs.month() > __rhs.month()) return false;
    return __lhs.day() < __rhs.day();
}

_LIBCPP_HIDE_FROM_ABI inline constexpr
bool operator> (const year_month_day& __lhs, const year_month_day& __rhs) noexcept
{ return __rhs < __lhs; }

_LIBCPP_HIDE_FROM_ABI inline constexpr
bool operator<=(const year_month_day& __lhs, const year_month_day& __rhs) noexcept
{ return !(__rhs < __lhs);}

_LIBCPP_HIDE_FROM_ABI inline constexpr
bool operator>=(const year_month_day& __lhs, const year_month_day& __rhs) noexcept
{ return !(__lhs < __rhs); }

_LIBCPP_HIDE_FROM_ABI inline constexpr
year_month_day operator/(const year_month& __lhs, const day& __rhs) noexcept
{ return year_month_day{__lhs.year(), __lhs.month(), __rhs}; }

_LIBCPP_HIDE_FROM_ABI inline constexpr
year_month_day operator/(const year_month& __lhs, int __rhs) noexcept
{ return __lhs / day(__rhs); }

_LIBCPP_HIDE_FROM_ABI inline constexpr
year_month_day operator/(const year& __lhs, const month_day& __rhs) noexcept
{ return __lhs / __rhs.month() / __rhs.day(); }

_LIBCPP_HIDE_FROM_ABI inline constexpr
year_month_day operator/(int __lhs, const month_day& __rhs) noexcept
{ return year(__lhs) / __rhs; }

_LIBCPP_HIDE_FROM_ABI inline constexpr
year_month_day operator/(const month_day& __lhs, const year& __rhs) noexcept
{ return __rhs / __lhs; }

_LIBCPP_HIDE_FROM_ABI inline constexpr
year_month_day operator/(const month_day& __lhs, int __rhs) noexcept
{ return year(__rhs) / __lhs; }


_LIBCPP_HIDE_FROM_ABI inline constexpr
year_month_day operator+(const year_month_day& __lhs, const months& __rhs) noexcept
{ return (__lhs.year()/__lhs.month() + __rhs)/__lhs.day(); }

_LIBCPP_HIDE_FROM_ABI inline constexpr
year_month_day operator+(const months& __lhs, const year_month_day& __rhs) noexcept
{ return __rhs + __lhs; }

_LIBCPP_HIDE_FROM_ABI inline constexpr
year_month_day operator-(const year_month_day& __lhs, const months& __rhs) noexcept
{ return __lhs + -__rhs; }

_LIBCPP_HIDE_FROM_ABI inline constexpr
year_month_day operator+(const year_month_day& __lhs, const years& __rhs) noexcept
{ return (__lhs.year() + __rhs) / __lhs.month() / __lhs.day(); }

_LIBCPP_HIDE_FROM_ABI inline constexpr
year_month_day operator+(const years& __lhs, const year_month_day& __rhs) noexcept
{ return __rhs + __lhs; }

_LIBCPP_HIDE_FROM_ABI inline constexpr
year_month_day operator-(const year_month_day& __lhs, const years& __rhs) noexcept
{ return __lhs + -__rhs; }

_LIBCPP_HIDE_FROM_ABI inline constexpr year_month_day& year_month_day::operator+=(const months& __dm) noexcept { *this = *this + __dm; return *this; }
_LIBCPP_HIDE_FROM_ABI inline constexpr year_month_day& year_month_day::operator-=(const months& __dm) noexcept { *this = *this - __dm; return *this; }
_LIBCPP_HIDE_FROM_ABI inline constexpr year_month_day& year_month_day::operator+=(const years& __dy)  noexcept { *this = *this + __dy; return *this; }
_LIBCPP_HIDE_FROM_ABI inline constexpr year_month_day& year_month_day::operator-=(const years& __dy)  noexcept { *this = *this - __dy; return *this; }

class year_month_day_last {
private:
    chrono::year           __y;
    chrono::month_day_last __mdl;
public:
     _LIBCPP_HIDE_FROM_ABI constexpr year_month_day_last(const year& __yval, const month_day_last& __mdlval) noexcept
        : __y{__yval}, __mdl{__mdlval} {}

     _LIBCPP_HIDE_FROM_ABI constexpr year_month_day_last& operator+=(const months& __m) noexcept;
     _LIBCPP_HIDE_FROM_ABI constexpr year_month_day_last& operator-=(const months& __m) noexcept;
     _LIBCPP_HIDE_FROM_ABI constexpr year_month_day_last& operator+=(const years& __y)  noexcept;
     _LIBCPP_HIDE_FROM_ABI constexpr year_month_day_last& operator-=(const years& __y)  noexcept;

     _LIBCPP_HIDE_FROM_ABI inline constexpr chrono::year                     year() const noexcept { return __y; }
     _LIBCPP_HIDE_FROM_ABI inline constexpr chrono::month                   month() const noexcept { return __mdl.month(); }
     _LIBCPP_HIDE_FROM_ABI inline constexpr chrono::month_day_last month_day_last() const noexcept { return __mdl; }
     _LIBCPP_HIDE_FROM_ABI        constexpr chrono::day                       day() const noexcept;
     _LIBCPP_HIDE_FROM_ABI inline constexpr operator                     sys_days() const noexcept { return   sys_days{year()/month()/day()}; }
     _LIBCPP_HIDE_FROM_ABI inline explicit constexpr operator          local_days() const noexcept { return local_days{year()/month()/day()}; }
     _LIBCPP_HIDE_FROM_ABI inline constexpr bool                               ok() const noexcept { return __y.ok() && __mdl.ok(); }
};

_LIBCPP_HIDE_FROM_ABI inline constexpr
chrono::day year_month_day_last::day() const noexcept
{
    constexpr chrono::day __d[] =
    {
        chrono::day(31), chrono::day(28), chrono::day(31),
        chrono::day(30), chrono::day(31), chrono::day(30),
        chrono::day(31), chrono::day(31), chrono::day(30),
        chrono::day(31), chrono::day(30), chrono::day(31)
    };
    return (month() != February || !__y.is_leap()) && month().ok() ?
        __d[static_cast<unsigned>(month()) - 1] : chrono::day{29};
}

_LIBCPP_HIDE_FROM_ABI inline constexpr
bool operator==(const year_month_day_last& __lhs, const year_month_day_last& __rhs) noexcept
{ return __lhs.year() == __rhs.year() && __lhs.month_day_last() == __rhs.month_day_last(); }

_LIBCPP_HIDE_FROM_ABI inline constexpr
bool operator!=(const year_month_day_last& __lhs, const year_month_day_last& __rhs) noexcept
{ return !(__lhs == __rhs); }

_LIBCPP_HIDE_FROM_ABI inline constexpr
bool operator< (const year_month_day_last& __lhs, const year_month_day_last& __rhs) noexcept
{
    if (__lhs.year() < __rhs.year()) return true;
    if (__lhs.year() > __rhs.year()) return false;
    return __lhs.month_day_last() < __rhs.month_day_last();
}

_LIBCPP_HIDE_FROM_ABI inline constexpr
bool operator> (const year_month_day_last& __lhs, const year_month_day_last& __rhs) noexcept
{ return __rhs < __lhs; }

_LIBCPP_HIDE_FROM_ABI inline constexpr
bool operator<=(const year_month_day_last& __lhs, const year_month_day_last& __rhs) noexcept
{ return !(__rhs < __lhs);}

_LIBCPP_HIDE_FROM_ABI inline constexpr
bool operator>=(const year_month_day_last& __lhs, const year_month_day_last& __rhs) noexcept
{ return !(__lhs < __rhs); }

_LIBCPP_HIDE_FROM_ABI inline constexpr
year_month_day_last operator/(const year_month& __lhs, last_spec) noexcept
{ return year_month_day_last{__lhs.year(), month_day_last{__lhs.month()}}; }

_LIBCPP_HIDE_FROM_ABI inline constexpr
year_month_day_last operator/(const year& __lhs, const month_day_last& __rhs) noexcept
{ return year_month_day_last{__lhs, __rhs}; }

_LIBCPP_HIDE_FROM_ABI inline constexpr
year_month_day_last operator/(int __lhs, const month_day_last& __rhs) noexcept
{ return year_month_day_last{year{__lhs}, __rhs}; }

_LIBCPP_HIDE_FROM_ABI inline constexpr year_month_day_last
operator/(const month_day_last& __lhs, const year& __rhs) noexcept
{ return __rhs / __lhs; }

_LIBCPP_HIDE_FROM_ABI inline constexpr
year_month_day_last operator/(const month_day_last& __lhs, int __rhs) noexcept
{ return year{__rhs} / __lhs; }


_LIBCPP_HIDE_FROM_ABI inline constexpr
year_month_day_last operator+(const year_month_day_last& __lhs, const months& __rhs) noexcept
{ return (__lhs.year() / __lhs.month() + __rhs) / last; }

_LIBCPP_HIDE_FROM_ABI inline constexpr
year_month_day_last operator+(const months& __lhs, const year_month_day_last& __rhs) noexcept
{ return __rhs + __lhs; }

_LIBCPP_HIDE_FROM_ABI inline constexpr
year_month_day_last operator-(const year_month_day_last& __lhs, const months& __rhs) noexcept
{ return __lhs + (-__rhs); }

_LIBCPP_HIDE_FROM_ABI inline constexpr
year_month_day_last operator+(const year_month_day_last& __lhs, const years& __rhs) noexcept
{ return year_month_day_last{__lhs.year() + __rhs, __lhs.month_day_last()}; }

_LIBCPP_HIDE_FROM_ABI inline constexpr
year_month_day_last operator+(const years& __lhs, const year_month_day_last& __rhs) noexcept
{ return __rhs + __lhs; }

_LIBCPP_HIDE_FROM_ABI inline constexpr
year_month_day_last operator-(const year_month_day_last& __lhs, const years& __rhs) noexcept
{ return __lhs + (-__rhs); }

_LIBCPP_HIDE_FROM_ABI inline constexpr year_month_day_last& year_month_day_last::operator+=(const months& __dm) noexcept { *this = *this + __dm; return *this; }
_LIBCPP_HIDE_FROM_ABI inline constexpr year_month_day_last& year_month_day_last::operator-=(const months& __dm) noexcept { *this = *this - __dm; return *this; }
_LIBCPP_HIDE_FROM_ABI inline constexpr year_month_day_last& year_month_day_last::operator+=(const years& __dy)  noexcept { *this = *this + __dy; return *this; }
_LIBCPP_HIDE_FROM_ABI inline constexpr year_month_day_last& year_month_day_last::operator-=(const years& __dy)  noexcept { *this = *this - __dy; return *this; }

_LIBCPP_HIDE_FROM_ABI inline constexpr
year_month_day::year_month_day(const year_month_day_last& __ymdl) noexcept
    : __y{__ymdl.year()}, __m{__ymdl.month()}, __d{__ymdl.day()} {}

_LIBCPP_HIDE_FROM_ABI inline constexpr
bool year_month_day::ok() const noexcept
{
    if (!__y.ok() || !__m.ok()) return false;
    return chrono::day{1} <= __d && __d <= (__y / __m / last).day();
}

class year_month_weekday {
    chrono::year            __y;
    chrono::month           __m;
    chrono::weekday_indexed __wdi;
public:
    _LIBCPP_HIDE_FROM_ABI year_month_weekday() = default;
    _LIBCPP_HIDE_FROM_ABI constexpr year_month_weekday(const chrono::year& __yval, const chrono::month& __mval,
                               const chrono::weekday_indexed& __wdival) noexcept
        : __y{__yval}, __m{__mval}, __wdi{__wdival} {}
    _LIBCPP_HIDE_FROM_ABI constexpr year_month_weekday(const sys_days& __sysd) noexcept
            : year_month_weekday(__from_days(__sysd.time_since_epoch())) {}
    _LIBCPP_HIDE_FROM_ABI inline explicit constexpr year_month_weekday(const local_days& __locd) noexcept
            : year_month_weekday(__from_days(__locd.time_since_epoch())) {}
    _LIBCPP_HIDE_FROM_ABI constexpr year_month_weekday& operator+=(const months& m) noexcept;
    _LIBCPP_HIDE_FROM_ABI constexpr year_month_weekday& operator-=(const months& m) noexcept;
    _LIBCPP_HIDE_FROM_ABI constexpr year_month_weekday& operator+=(const years& y)  noexcept;
    _LIBCPP_HIDE_FROM_ABI constexpr year_month_weekday& operator-=(const years& y)  noexcept;

    _LIBCPP_HIDE_FROM_ABI inline constexpr chrono::year                       year() const noexcept { return __y; }
    _LIBCPP_HIDE_FROM_ABI inline constexpr chrono::month                     month() const noexcept { return __m; }
    _LIBCPP_HIDE_FROM_ABI inline constexpr chrono::weekday                 weekday() const noexcept { return __wdi.weekday(); }
    _LIBCPP_HIDE_FROM_ABI inline constexpr unsigned                          index() const noexcept { return __wdi.index(); }
    _LIBCPP_HIDE_FROM_ABI inline constexpr chrono::weekday_indexed weekday_indexed() const noexcept { return __wdi; }

    _LIBCPP_HIDE_FROM_ABI inline constexpr                       operator sys_days() const noexcept { return   sys_days{__to_days()}; }
    _LIBCPP_HIDE_FROM_ABI inline explicit constexpr operator            local_days() const noexcept { return local_days{__to_days()}; }
    _LIBCPP_HIDE_FROM_ABI inline constexpr bool ok() const noexcept
    {
        if (!__y.ok() || !__m.ok() || !__wdi.ok()) return false;
        if (__wdi.index() <= 4) return true;
        auto __nth_weekday_day =
            __wdi.weekday() -
            chrono::weekday{static_cast<sys_days>(__y / __m / 1)} +
            days{(__wdi.index() - 1) * 7 + 1};
        return static_cast<unsigned>(__nth_weekday_day.count()) <=
               static_cast<unsigned>((__y / __m / last).day());
    }

    _LIBCPP_HIDE_FROM_ABI static constexpr year_month_weekday __from_days(days __d) noexcept;
    _LIBCPP_HIDE_FROM_ABI constexpr days __to_days() const noexcept;
};

_LIBCPP_HIDE_FROM_ABI inline constexpr
year_month_weekday year_month_weekday::__from_days(days __d) noexcept
{
    const sys_days      __sysd{__d};
    const chrono::weekday __wd = chrono::weekday(__sysd);
    const year_month_day __ymd = year_month_day(__sysd);
    return year_month_weekday{__ymd.year(), __ymd.month(),
                              __wd[(static_cast<unsigned>(__ymd.day())-1)/7+1]};
}

_LIBCPP_HIDE_FROM_ABI inline constexpr
days year_month_weekday::__to_days() const noexcept
{
    const sys_days __sysd = sys_days(__y/__m/1);
    return (__sysd + (__wdi.weekday() - chrono::weekday(__sysd) + days{(__wdi.index()-1)*7}))
                .time_since_epoch();
}

_LIBCPP_HIDE_FROM_ABI inline constexpr
bool operator==(const year_month_weekday& __lhs, const year_month_weekday& __rhs) noexcept
{ return __lhs.year() == __rhs.year() && __lhs.month() == __rhs.month() && __lhs.weekday_indexed() == __rhs.weekday_indexed(); }

_LIBCPP_HIDE_FROM_ABI inline constexpr
bool operator!=(const year_month_weekday& __lhs, const year_month_weekday& __rhs) noexcept
{ return !(__lhs == __rhs); }

_LIBCPP_HIDE_FROM_ABI inline constexpr
year_month_weekday operator/(const year_month& __lhs, const weekday_indexed& __rhs) noexcept
{ return year_month_weekday{__lhs.year(), __lhs.month(), __rhs}; }

_LIBCPP_HIDE_FROM_ABI inline constexpr
year_month_weekday operator/(const year& __lhs, const month_weekday& __rhs) noexcept
{ return year_month_weekday{__lhs, __rhs.month(), __rhs.weekday_indexed()}; }

_LIBCPP_HIDE_FROM_ABI inline constexpr
year_month_weekday operator/(int __lhs, const month_weekday& __rhs) noexcept
{ return year(__lhs) / __rhs; }

_LIBCPP_HIDE_FROM_ABI inline constexpr
year_month_weekday operator/(const month_weekday& __lhs, const year& __rhs) noexcept
{ return __rhs / __lhs; }

_LIBCPP_HIDE_FROM_ABI inline constexpr
year_month_weekday operator/(const month_weekday& __lhs, int __rhs) noexcept
{ return year(__rhs) / __lhs; }


_LIBCPP_HIDE_FROM_ABI inline constexpr
year_month_weekday operator+(const year_month_weekday& __lhs, const months& __rhs) noexcept
{ return (__lhs.year() / __lhs.month() + __rhs) / __lhs.weekday_indexed(); }

_LIBCPP_HIDE_FROM_ABI inline constexpr
year_month_weekday operator+(const months& __lhs, const year_month_weekday& __rhs) noexcept
{ return __rhs + __lhs; }

_LIBCPP_HIDE_FROM_ABI inline constexpr
year_month_weekday operator-(const year_month_weekday& __lhs, const months& __rhs) noexcept
{ return __lhs + (-__rhs); }

_LIBCPP_HIDE_FROM_ABI inline constexpr
year_month_weekday operator+(const year_month_weekday& __lhs, const years& __rhs) noexcept
{ return year_month_weekday{__lhs.year() + __rhs, __lhs.month(), __lhs.weekday_indexed()}; }

_LIBCPP_HIDE_FROM_ABI inline constexpr
year_month_weekday operator+(const years& __lhs, const year_month_weekday& __rhs) noexcept
{ return __rhs + __lhs; }

_LIBCPP_HIDE_FROM_ABI inline constexpr
year_month_weekday operator-(const year_month_weekday& __lhs, const years& __rhs) noexcept
{ return __lhs + (-__rhs); }


_LIBCPP_HIDE_FROM_ABI inline constexpr year_month_weekday& year_month_weekday::operator+=(const months& __dm) noexcept { *this = *this + __dm; return *this; }
_LIBCPP_HIDE_FROM_ABI inline constexpr year_month_weekday& year_month_weekday::operator-=(const months& __dm) noexcept { *this = *this - __dm; return *this; }
_LIBCPP_HIDE_FROM_ABI inline constexpr year_month_weekday& year_month_weekday::operator+=(const years& __dy)  noexcept { *this = *this + __dy; return *this; }
_LIBCPP_HIDE_FROM_ABI inline constexpr year_month_weekday& year_month_weekday::operator-=(const years& __dy)  noexcept { *this = *this - __dy; return *this; }

class year_month_weekday_last {
private:
    chrono::year         __y;
    chrono::month        __m;
    chrono::weekday_last __wdl;
public:
    _LIBCPP_HIDE_FROM_ABI constexpr year_month_weekday_last(const chrono::year& __yval, const chrono::month& __mval,
                                      const chrono::weekday_last& __wdlval) noexcept
                : __y{__yval}, __m{__mval}, __wdl{__wdlval} {}
    _LIBCPP_HIDE_FROM_ABI constexpr year_month_weekday_last& operator+=(const months& __dm) noexcept;
    _LIBCPP_HIDE_FROM_ABI constexpr year_month_weekday_last& operator-=(const months& __dm) noexcept;
    _LIBCPP_HIDE_FROM_ABI constexpr year_month_weekday_last& operator+=(const years& __dy)  noexcept;
    _LIBCPP_HIDE_FROM_ABI constexpr year_month_weekday_last& operator-=(const years& __dy)  noexcept;

    _LIBCPP_HIDE_FROM_ABI inline constexpr chrono::year                 year() const noexcept { return __y; }
    _LIBCPP_HIDE_FROM_ABI inline constexpr chrono::month               month() const noexcept { return __m; }
    _LIBCPP_HIDE_FROM_ABI inline constexpr chrono::weekday           weekday() const noexcept { return __wdl.weekday(); }
    _LIBCPP_HIDE_FROM_ABI inline constexpr chrono::weekday_last weekday_last() const noexcept { return __wdl; }
    _LIBCPP_HIDE_FROM_ABI inline constexpr operator                 sys_days() const noexcept { return   sys_days{__to_days()}; }
    _LIBCPP_HIDE_FROM_ABI inline explicit constexpr operator      local_days() const noexcept { return local_days{__to_days()}; }
    _LIBCPP_HIDE_FROM_ABI inline constexpr bool ok() const noexcept { return __y.ok() && __m.ok() && __wdl.ok(); }

    _LIBCPP_HIDE_FROM_ABI constexpr days __to_days() const noexcept;

};

_LIBCPP_HIDE_FROM_ABI inline constexpr
days year_month_weekday_last::__to_days() const noexcept
{
    const sys_days __last = sys_days{__y/__m/last};
    return (__last - (chrono::weekday{__last} - __wdl.weekday())).time_since_epoch();

}

_LIBCPP_HIDE_FROM_ABI inline constexpr
bool operator==(const year_month_weekday_last& __lhs, const year_month_weekday_last& __rhs) noexcept
{ return __lhs.year() == __rhs.year() && __lhs.month() == __rhs.month() && __lhs.weekday_last() == __rhs.weekday_last(); }

_LIBCPP_HIDE_FROM_ABI inline constexpr
bool operator!=(const year_month_weekday_last& __lhs, const year_month_weekday_last& __rhs) noexcept
{ return !(__lhs == __rhs); }


_LIBCPP_HIDE_FROM_ABI inline constexpr
year_month_weekday_last operator/(const year_month& __lhs, const weekday_last& __rhs) noexcept
{ return year_month_weekday_last{__lhs.year(), __lhs.month(), __rhs}; }

_LIBCPP_HIDE_FROM_ABI inline constexpr
year_month_weekday_last operator/(const year& __lhs, const month_weekday_last& __rhs) noexcept
{ return year_month_weekday_last{__lhs, __rhs.month(), __rhs.weekday_last()}; }

_LIBCPP_HIDE_FROM_ABI inline constexpr
year_month_weekday_last operator/(int __lhs, const month_weekday_last& __rhs) noexcept
{ return year(__lhs) / __rhs; }

_LIBCPP_HIDE_FROM_ABI inline constexpr
year_month_weekday_last operator/(const month_weekday_last& __lhs, const year& __rhs) noexcept
{ return __rhs / __lhs; }

_LIBCPP_HIDE_FROM_ABI inline constexpr
year_month_weekday_last operator/(const month_weekday_last& __lhs, int __rhs) noexcept
{ return year(__rhs) / __lhs; }


_LIBCPP_HIDE_FROM_ABI inline constexpr
year_month_weekday_last operator+(const year_month_weekday_last& __lhs, const months& __rhs) noexcept
{ return (__lhs.year() / __lhs.month() + __rhs) / __lhs.weekday_last(); }

_LIBCPP_HIDE_FROM_ABI inline constexpr
year_month_weekday_last operator+(const months& __lhs, const year_month_weekday_last& __rhs) noexcept
{ return __rhs + __lhs; }

_LIBCPP_HIDE_FROM_ABI inline constexpr
year_month_weekday_last operator-(const year_month_weekday_last& __lhs, const months& __rhs) noexcept
{ return __lhs + (-__rhs); }

_LIBCPP_HIDE_FROM_ABI inline constexpr
year_month_weekday_last operator+(const year_month_weekday_last& __lhs, const years& __rhs) noexcept
{ return year_month_weekday_last{__lhs.year() + __rhs, __lhs.month(), __lhs.weekday_last()}; }

_LIBCPP_HIDE_FROM_ABI inline constexpr
year_month_weekday_last operator+(const years& __lhs, const year_month_weekday_last& __rhs) noexcept
{ return __rhs + __lhs; }

_LIBCPP_HIDE_FROM_ABI inline constexpr
year_month_weekday_last operator-(const year_month_weekday_last& __lhs, const years& __rhs) noexcept
{ return __lhs + (-__rhs); }

_LIBCPP_HIDE_FROM_ABI inline constexpr year_month_weekday_last& year_month_weekday_last::operator+=(const months& __dm) noexcept { *this = *this + __dm; return *this; }
_LIBCPP_HIDE_FROM_ABI inline constexpr year_month_weekday_last& year_month_weekday_last::operator-=(const months& __dm) noexcept { *this = *this - __dm; return *this; }
_LIBCPP_HIDE_FROM_ABI inline constexpr year_month_weekday_last& year_month_weekday_last::operator+=(const years& __dy)  noexcept { *this = *this + __dy; return *this; }
_LIBCPP_HIDE_FROM_ABI inline constexpr year_month_weekday_last& year_month_weekday_last::operator-=(const years& __dy)  noexcept { *this = *this - __dy; return *this; }


template <class _Duration>
class hh_mm_ss
{
private:
    static_assert(__is_duration<_Duration>::value, "template parameter of hh_mm_ss must be a std::chrono::duration");
    using __CommonType = common_type_t<_Duration, chrono::seconds>;

    _LIBCPP_HIDE_FROM_ABI static constexpr uint64_t __pow10(unsigned __exp)
    {
        uint64_t __ret = 1;
        for (unsigned __i = 0; __i < __exp; ++__i)
            __ret *= 10U;
        return __ret;
    }

    _LIBCPP_HIDE_FROM_ABI static constexpr unsigned __width(uint64_t __n, uint64_t __d = 10, unsigned __w = 0)
    {
        if (__n >= 2 && __d != 0 && __w < 19)
            return 1 + __width(__n, __d % __n * 10, __w+1);
        return 0;
    }

public:
    _LIBCPP_HIDE_FROM_ABI static unsigned constexpr fractional_width = __width(__CommonType::period::den) < 19 ?
                                                 __width(__CommonType::period::den) : 6u;
    using precision = duration<typename __CommonType::rep, ratio<1, __pow10(fractional_width)>>;

    _LIBCPP_HIDE_FROM_ABI constexpr hh_mm_ss() noexcept : hh_mm_ss{_Duration::zero()} {}

    _LIBCPP_HIDE_FROM_ABI constexpr explicit hh_mm_ss(_Duration __d) noexcept :
        __is_neg(__d < _Duration(0)),
        __h(duration_cast<chrono::hours>  (abs(__d))),
        __m(duration_cast<chrono::minutes>(abs(__d) - hours())),
        __s(duration_cast<chrono::seconds>(abs(__d) - hours() - minutes())),
        __f(duration_cast<precision>      (abs(__d) - hours() - minutes() - seconds()))
        {}

    _LIBCPP_HIDE_FROM_ABI constexpr bool is_negative()        const noexcept { return __is_neg; }
    _LIBCPP_HIDE_FROM_ABI constexpr chrono::hours hours()     const noexcept { return __h; }
    _LIBCPP_HIDE_FROM_ABI constexpr chrono::minutes minutes() const noexcept { return __m; }
    _LIBCPP_HIDE_FROM_ABI constexpr chrono::seconds seconds() const noexcept { return __s; }
    _LIBCPP_HIDE_FROM_ABI constexpr precision subseconds()    const noexcept { return __f; }

    _LIBCPP_HIDE_FROM_ABI constexpr precision to_duration() const noexcept
    {
        auto __dur = __h + __m + __s + __f;
        return __is_neg ? -__dur : __dur;
    }

    _LIBCPP_HIDE_FROM_ABI constexpr explicit operator precision() const noexcept { return to_duration(); }

private:
    bool            __is_neg;
    chrono::hours   __h;
    chrono::minutes __m;
    chrono::seconds __s;
    precision       __f;
};

_LIBCPP_HIDE_FROM_ABI constexpr bool is_am(const hours& __h) noexcept { return __h >= hours( 0) && __h < hours(12); }
_LIBCPP_HIDE_FROM_ABI constexpr bool is_pm(const hours& __h) noexcept { return __h >= hours(12) && __h < hours(24); }

_LIBCPP_HIDE_FROM_ABI constexpr hours make12(const hours& __h) noexcept
{
    if      (__h == hours( 0)) return hours(12);
    else if (__h <= hours(12)) return __h;
    else                       return __h - hours(12);
}

_LIBCPP_HIDE_FROM_ABI constexpr hours make24(const hours& __h, bool __is_pm) noexcept
{
    if (__is_pm)
        return __h == hours(12) ? __h : __h + hours(12);
    else
        return __h == hours(12) ? hours(0) : __h;
}

} // namespace chrono

inline namespace literals
{
  inline namespace chrono_literals
  {
    _LIBCPP_HIDE_FROM_ABI constexpr chrono::day operator ""d(unsigned long long __d) noexcept
    {
        return chrono::day(static_cast<unsigned>(__d));
    }

    _LIBCPP_HIDE_FROM_ABI constexpr chrono::year operator ""y(unsigned long long __y) noexcept
    {
        return chrono::year(static_cast<int>(__y));
    }
} // namespace chrono_literals
} // namespace literals

namespace chrono { // hoist the literals into namespace std::chrono
   using namespace literals::chrono_literals;
} // namespace chrono

_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP_STD_VER > 17

_LIBCPP_POP_MACROS

#endif // _LIBCPP___CHRONO_CALENDAR_H
