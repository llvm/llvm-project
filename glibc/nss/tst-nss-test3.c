/* Test error checking for group entries.
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

/* The names here are arbitrary, but the *lengths* of the arrays is
   not, and groups 6 and 7 test for partial matches.  */

static const char *group_2[] = {
  "foo", "bar", NULL
};

static const char *group_3[] = {
  "tom", "dick", "harry", NULL
};

static const char *group_4[] = {
  "alpha", "beta", "gamma", "fred", NULL
};

static const char *group_6[] = {
  "larry", "curly", "moe", NULL
};

static const char *group_7[] = {
  "larry", "curly", "darryl", NULL
};

static const char *group_14[] = {
  "huey", "dewey", "louis", NULL
};

/* Note that we're intentionally causing mis-matches here; the purpose
   of this test case is to test each error check and make sure they
   detect the errors they check for, and to ensure that the harness
   can process all the error cases properly (i.e. a NULL gr_name
   field).  We check for the correct number of mismatches at the
   end.  */

/* This is the data we're giving the service.  */
static struct group group_table_data[] = {
  GRP(4), /* match */
  GRP_N(8, "name6", group_6), /* wrong gid */
  GRP_N(14, NULL, group_14), /* missing name */
  GRP(14), /* unexpected name */
  GRP_N(7, "name7_wrong", group_7), /* wrong name */
  { .gr_name =  (char *)"name5", .gr_passwd =  (char *)"wilma", .gr_gid = 5, .gr_mem = NULL }, /* unexpected passwd */
  { .gr_name =  (char *)"name5", .gr_passwd = NULL, .gr_gid = 5, .gr_mem = NULL }, /* missing passwd */
  { .gr_name =  (char *)"name5", .gr_passwd = (char *)"wilma", .gr_gid = 5, .gr_mem = NULL }, /* wrong passwd */
  GRP_N(3, "name3a", NULL),   /* missing member list */
  GRP_N(3, "name3b", group_3), /* unexpected member list */
  GRP_N(3, "name3c", group_3), /* wrong/short member list */
  GRP_N(3, "name3d", group_4), /* wrong/long member list */
  GRP_LAST ()
};

/* This is the data we compare against.  */
static struct group group_table[] = {
  GRP(4),
  GRP(6),
  GRP(14),
  GRP_N(14, NULL, group_14),
  GRP(7),
  { .gr_name =  (char *)"name5", .gr_passwd = NULL, .gr_gid = 5, .gr_mem = NULL },
  { .gr_name =  (char *)"name5", .gr_passwd =  (char *)"fred", .gr_gid = 5, .gr_mem = NULL },
  { .gr_name =  (char *)"name5", .gr_passwd =  (char *)"fred", .gr_gid = 5, .gr_mem = NULL },
  GRP_N(3, "name3a", group_3),
  GRP_N(3, "name3b", NULL),
  GRP_N(3, "name3c", group_4),
  GRP_N(3, "name3d", group_3),
  GRP(2),
  GRP_LAST ()
};

void
_nss_test1_init_hook(test_tables *t)
{
  t->grp_table = group_table_data;
}

static int
do_test (void)
{
  int retval = 0;
  int i;
  struct group *g = NULL;

/* Previously we used __nss_configure_lookup to isolate the test
   from the host environment and to get it to lookup from our new
   test1 NSS service module, but now this test is run in a different
   root filesystem via the test-container support and we directly
   configure the use of the test1 NSS service.  */

  setgrent ();

  i = 0;
  for (g = getgrent () ;
       g != NULL && ! GRP_ISLAST(&group_table[i]);
       ++i, g = getgrent ())
    {
      retval += compare_groups (i, g, & group_table[i]);
    }

  endgrent ();

  if (g)
    {
      printf ("FAIL: [?] group entry %u.%s unexpected\n", g->gr_gid, g->gr_name);
      ++retval;
    }
  if (group_table[i].gr_name || group_table[i].gr_gid)
    {
      printf ("FAIL: [%d] group entry %u.%s missing\n", i,
	      group_table[i].gr_gid, group_table[i].gr_name);
      ++retval;
    }

#define EXPECTED 18
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
