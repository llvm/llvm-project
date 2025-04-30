/* Test that initgroups works.
   Copyright (C) 2020-2021 Free Software Foundation, Inc.
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

#include <nss.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <errno.h>
#include <unistd.h>
#include <grp.h>

#include <support/support.h>
#include <support/check.h>

/* Test that initgroups includes secondary groups.
   https://bugzilla.redhat.com/show_bug.cgi?id=1906066  */

/* This version uses the wrapper around the groups module.  */

#define EXPECTED_N_GROUPS 4
static gid_t expected_groups[] =
  { 20, 30, 50, 51 };

static int
do_test (void)
{
  gid_t mygroups [50];
  int i, n;

  n = 50;
  getgrouplist ("dj", 20, mygroups, &n);

  TEST_COMPARE (n, EXPECTED_N_GROUPS);
  for (i=0; i<n; i++)
    TEST_COMPARE (mygroups[i], expected_groups[i]);

  return 0;
}

#include <support/test-driver.c>
