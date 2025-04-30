/* elision-timed.c: Lock elision timed lock.
   Copyright (C) 2015-2021 Free Software Foundation, Inc.
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

#include <time.h>
#include <elision-conf.h>
#include "lowlevellock.h"
#include "futex-internal.h"

#define __lll_lock_elision __lll_clocklock_elision
#define EXTRAARG clockid_t clockid, const struct __timespec64 *t,
#undef LLL_LOCK
#define LLL_LOCK(a, b) __futex_clocklock64 (&(a), clockid, t, b)

#include "elision-lock.c"
