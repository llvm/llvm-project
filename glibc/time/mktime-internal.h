/* Internals of mktime and related functions
   Copyright 2016-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Paul Eggert <eggert@cs.ucla.edu>.

   The GNU C Library is free software; you can redistribute it and/or
   modify it under the terms of the GNU Lesser General Public
   License as published by the Free Software Foundation; either
   version 2.1 of the License, or (at your option) any later version.

   The GNU C Library is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
   Lesser General Public License for more details.

   You should have received a copy of the GNU Lesser General Public
   License along with the GNU C Library; if not, see
   <https://www.gnu.org/licenses/>.  */

#ifndef _LIBC
# include <time.h>
#endif

/* mktime_offset_t is a signed type wide enough to hold a UTC offset
   in seconds, and used as part of the type of the offset-guess
   argument to mktime_internal.  In Glibc, it is always long int.
   When in Gnulib, use time_t on platforms where time_t
   is signed, to be compatible with platforms like BeOS that export
   this implementation detail of mktime.  On platforms where time_t is
   unsigned, GNU and POSIX code can assume 'int' is at least 32 bits
   which is wide enough for a UTC offset.  */
#ifdef _LIBC
typedef long int mktime_offset_t;
#elif defined TIME_T_IS_SIGNED
typedef time_t mktime_offset_t;
#else
typedef int mktime_offset_t;
#endif

/* The source code uses identifiers like __time64_t for glibc
   timestamps that can contain 64-bit values even when time_t is only
   32 bits.  These are just macros for the ordinary identifiers unless
   compiling within glibc when time_t is 32 bits.  */
#if ! (defined _LIBC && __TIMESIZE != 64)
# undef __time64_t
# define __time64_t time_t
# define __gmtime64_r __gmtime_r
# define __localtime64_r __localtime_r
# define __mktime64 mktime
# define __timegm64 timegm
#endif

#ifndef _LIBC

/* Although glibc source code uses leading underscores, Gnulib wants
   ordinary names.

   Portable standalone applications should supply a <time.h> that
   declares a POSIX-compliant localtime_r, for the benefit of older
   implementations that lack localtime_r or have a nonstandard one.
   Similarly for gmtime_r.  See the gnulib time_r module for one way
   to implement this.  */

# undef __gmtime_r
# undef __localtime_r
# define __gmtime_r gmtime_r
# define __localtime_r localtime_r

# define __mktime_internal mktime_internal

#endif

/* Subroutine of mktime.  Return the time_t representation of TP and
   normalize TP, given that a struct tm * maps to a time_t as performed
   by FUNC.  Record next guess for localtime-gmtime offset in *OFFSET.  */
extern __time64_t __mktime_internal (struct tm *tp,
                                     struct tm *(*func) (__time64_t const *,
                                                         struct tm *),
                                     mktime_offset_t *offset) attribute_hidden;
