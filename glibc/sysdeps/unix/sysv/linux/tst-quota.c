/* Copyright (C) 2016-2021 Free Software Foundation, Inc.
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

#include <sys/quota.h>

#include <stdbool.h>
#include <stdio.h>

static bool errors;

void
check_size (const char *name1, size_t size1,
            const char *name2, size_t size2)
{
  const char *prefix;
  const char *op;
  if (size1 != size2)
    {
      prefix = "error";
      op = "!=";
      errors = true;
    }
  else
    {
      prefix = "info";
      op = "==";
    }
  printf ("%s: sizeof (%s) [%zu] %s sizeof (%s) [%zu]\n",
          prefix, name1, size1, op, name2, size2);
}

#define CHECK_SIZE(type1, type2) \
  check_size (#type1, sizeof (type1), #type2, sizeof (type2))

int
do_test (void)
{
  CHECK_SIZE (struct if_dqblk, struct dqblk);
  CHECK_SIZE (struct if_dqinfo, struct dqinfo);
  return errors;
}

#define TEST_FUNCTION do_test ()
#include "../test-skeleton.c"
