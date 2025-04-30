/* Test group merging.
   Copyright (C) 2017-2021 Free Software Foundation, Inc.
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

#include <support/support.h>

#include "nss_test.h"

/* The name choices here are arbitrary, aside from the merge_1 list
   needing to be an expected merge of group_1 and group_2.  */

static const char *group_1[] = {
  "foo", "bar", NULL
};

static const char *group_2[] = {
  "foo", "dick", "harry", NULL
};

/* Note that deduplication is NOT supposed to happen.  */
static const char *merge_1[] = {
  "foo", "bar", "foo", "dick", "harry", NULL
};

static const char *group_4[] = {
  "fred", "wilma", NULL
};

/* This is the data we're giving the service.  */
static struct group group_table_data1[] = {
  GRP_N(1, "name1", group_1),
  GRP(2),
  GRP_LAST ()
};

/* This is the data we're giving the service.  */
static struct group group_table_data2[] = {
  GRP_N(1, "name1", group_2),
  GRP(4),
  GRP_LAST ()
};

/* This is the data we compare against.  */
static struct group group_table[] = {
  GRP_N(1, "name1", merge_1),
  GRP(2),
  GRP(4),
  GRP_LAST ()
};

void
_nss_test1_init_hook(test_tables *t)
{
  t->grp_table = group_table_data1;
}

void
_nss_test2_init_hook(test_tables *t)
{
  t->grp_table = group_table_data2;
}

static int
do_test (void)
{
  int retval = 0;
  int i;
  struct group *g = NULL;
  uintptr_t align_mask;

  __nss_configure_lookup ("group", "test1 [SUCCESS=merge] test2");

  align_mask = __alignof__ (struct group *) - 1;

  setgrent ();

  for (i = 0; group_table[i].gr_gid; ++i)
    {
      g = getgrgid (group_table[i].gr_gid);
      if (g)
	{
	  retval += compare_groups (i, g, & group_table[i]);
	  if ((uintptr_t)g & align_mask)
	    {
	      printf("FAIL: [%d] unaligned group %p\n", i, g);
	      ++retval;
	    }
	  if ((uintptr_t)(g->gr_mem) & align_mask)
	    {
	      printf("FAIL: [%d] unaligned member list %p\n", i, g->gr_mem);
	      ++retval;
	    }
	}
      else
	{
	  printf ("FAIL: [%d] group %u.%s not found\n", i,
	      group_table[i].gr_gid, group_table[i].gr_name);
	  ++retval;
	}
    }

  endgrent ();

#define EXPECTED 0
  if (retval == EXPECTED)
    {
      if (retval > 0)
	printf ("PASS: Found %d expected errors\n", retval);
      return 0;
    }
  else
    {
      printf ("FAIL: Found %d errors, expected %d\n", retval, EXPECTED);
      return 1;
    }
}

#include <support/test-driver.c>
