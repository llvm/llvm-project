/* Single-threaded test for sched_getaffinity and sched_setaffinity.
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

/* Function definitions for the benefit of
   tst-skeleton-affinity.c.  */

#include <stdbool.h>
#include <sched.h>

static int
getaffinity (size_t size, cpu_set_t *set)
{
  return sched_getaffinity (0, size, set);
}

static int
setaffinity (size_t size, const cpu_set_t *set)
{
  return sched_setaffinity (0, size, set);
}

struct conf;
static bool early_test (struct conf *unused)
{
  return true;
}

#include "tst-skeleton-affinity.c"
