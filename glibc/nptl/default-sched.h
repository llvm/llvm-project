/* Determine calling thread's scheduling parameters.  Stub version.
   Copyright (C) 2014-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.

   The GNU C Library is free software; you can redistribute it and/or
   modify it under the terms of the GNU Library General Public License as
   published by the Free Software Foundation; either version 2 of the
   License, or (at your option) any later version.

   The GNU C Library is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
   Library General Public License for more details.

   You should have received a copy of the GNU Library General Public
   License along with the GNU C Library; see the file COPYING.LIB.  If
   not, see <https://www.gnu.org/licenses/>.  */

#include <assert.h>

/* This should fill in PD->schedpolicy if PD->flags does not contain
   ATTR_FLAG_POLICY_SET, and set it; and PD->schedparam if PD->flags does
   not contain ATTR_FLAG_SCHED_SET, and set it.  It won't be called at all
   if both bits are already set.  */

static void
collect_default_sched (struct pthread *pd)
{
  assert ((pd->flags & (ATTR_FLAG_SCHED_SET | ATTR_FLAG_POLICY_SET)) != 0);

  /* The generic/stub version is a no-op rather than just using the
     __sched_getscheduler and __sched_getparam functions so that there
     won't be stub warnings for those functions just because pthread_create
     was called without actually calling those.  */
}
