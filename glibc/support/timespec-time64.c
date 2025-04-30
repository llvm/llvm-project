/* Support code for timespec checks.  64-bit time support.
   Copyright (C) 2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.

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

#include <support/timespec.h>
#include <stdio.h>
#include <stdint.h>
#include <assert.h>
#include <intprops.h>

#if __TIMESIZE != 64
struct __timespec64 timespec_sub_time64 (struct __timespec64,
					 struct __timespec64);

#define test_timespec_before_impl          test_timespec_before_impl_time64
#define test_timespec_equal_or_after_impl  \
  test_timespec_equal_or_after_impl_time64
#define support_timespec_ns                support_timespec_ns_time64
#define support_timespec_normalize         support_timespec_normalize_time64
#define support_timespec_check_in_range    \
  support_timespec_check_in_range_time64
#define timespec                           __timespec64
#define timespec_sub                       timespec_sub_time64

#include "timespec.c"
#endif
