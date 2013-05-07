// -*- C++ -*-
//===----------------------- support/win32/support.h ----------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP_SUPPORT_WIN32_SUPPORT_H
#define _LIBCPP_SUPPORT_WIN32_SUPPORT_H

/*
   Functions and constants used in libc++ that are missing from the Windows C library.
  */

#include <__config>
#include <wchar.h>  // mbstate_t
#include <stdio.h> // _snwprintf
#define swprintf _snwprintf
#define vswprintf _vsnwprintf
#define vfscnaf fscanf

int vasprintf( char **sptr, const char *__restrict fmt , va_list ap );
int asprintf( char **sptr, const char *__restrict fmt, ...);
//int vfscanf( FILE *__restrict stream, const char *__restrict format,
//             va_list arg);

size_t mbsnrtowcs( wchar_t *__restrict dst, const char **__restrict src,
                   size_t nmc, size_t len, mbstate_t *__restrict ps );
size_t wcsnrtombs( char *__restrict dst, const wchar_t **__restrict src,
                   size_t nwc, size_t len, mbstate_t *__restrict ps );

#if defined(_MSC_VER)
#define snprintf _snprintf

#include <xlocinfo.h>
#define atoll _atoi64
#define strtoll _strtoi64
#define strtoull _strtoui64
#define wcstoll _wcstoi64
#define wcstoull _wcstoui64
_LIBCPP_ALWAYS_INLINE float strtof( const char *nptr, char **endptr )
{ return _Stof(nptr, endptr, 0); }
_LIBCPP_ALWAYS_INLINE double strtod( const char *nptr, char **endptr )
{ return _Stod(nptr, endptr, 0); }
_LIBCPP_ALWAYS_INLINE long double strtold( const char *nptr, char **endptr )
{ return _Stold(nptr, endptr, 0); }

#define _Exit _exit

#ifndef __clang__ // MSVC-based Clang also defines _MSC_VER
#include <intrin.h>

_LIBCPP_ALWAYS_INLINE int __builtin_popcount(unsigned int x) {
   static const unsigned int m1 = 0x55555555; //binary: 0101...
   static const unsigned int m2 = 0x33333333; //binary: 00110011..
   static const unsigned int m4 = 0x0f0f0f0f; //binary:  4 zeros,  4 ones ...
   static const unsigned int h01= 0x01010101; //the sum of 256 to the power of 0,1,2,3...
   x -= (x >> 1) & m1;             //put count of each 2 bits into those 2 bits
   x = (x & m2) + ((x >> 2) & m2); //put count of each 4 bits into those 4 bits
   x = (x + (x >> 4)) & m4;        //put count of each 8 bits into those 8 bits
   return (x * h01) >> 24;  //returns left 8 bits of x + (x<<8) + (x<<16) + (x<<24)
}

_LIBCPP_ALWAYS_INLINE int __builtin_popcountl(unsigned long x) {
  return __builtin_popcount(static_cast<int>(x));
}

_LIBCPP_ALWAYS_INLINE int __builtin_popcountll(unsigned long long x) {
   static const unsigned long long m1  = 0x5555555555555555; //binary: 0101...
   static const unsigned long long m2  = 0x3333333333333333; //binary: 00110011..
   static const unsigned long long m4  = 0x0f0f0f0f0f0f0f0f; //binary:  4 zeros,  4 ones ...
   static const unsigned long long h01 = 0x0101010101010101; //the sum of 256 to the power of 0,1,2,3...
   x -= (x >> 1) & m1;             //put count of each 2 bits into those 2 bits
   x = (x & m2) + ((x >> 2) & m2); //put count of each 4 bits into those 4 bits
   x = (x + (x >> 4)) & m4;        //put count of each 8 bits into those 8 bits
   return static_cast<int>((x * h01)>>56);  //returns left 8 bits of x + (x<<8) + (x<<16) + (x<<24) + ...
}

_LIBCPP_ALWAYS_INLINE int __builtin_ctz( unsigned int x )
{
   DWORD r = 0;
   _BitScanReverse(&r, x);
   return static_cast<int>(r);
}
// sizeof(long) == sizeof(int) on Windows
_LIBCPP_ALWAYS_INLINE int __builtin_ctzl( unsigned long x )
{ return __builtin_ctz( static_cast<int>(x) ); }
_LIBCPP_ALWAYS_INLINE int __builtin_ctzll( unsigned long long x )
{
    DWORD r = 0;
    _BitScanReverse64(&r, x);
    return static_cast<int>(r);
}
_LIBCPP_ALWAYS_INLINE int __builtin_clz( unsigned int x )
{
   DWORD r = 0;
   _BitScanForward(&r, x);
   return static_cast<int>(r);
}
// sizeof(long) == sizeof(int) on Windows
_LIBCPP_ALWAYS_INLINE int __builtin_clzl( unsigned long x )
{ return __builtin_clz( static_cast<int>(x) ); }
_LIBCPP_ALWAYS_INLINE int __builtin_clzll( unsigned long long x )
{
    DWORD r = 0;
    _BitScanForward64(&r, x);
    return static_cast<int>(r);
}
#endif // !__clang__
#endif // _MSC_VER

#endif // _LIBCPP_SUPPORT_WIN32_SUPPORT_H
