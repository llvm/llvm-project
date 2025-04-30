/* Test error checking for passwd entries.
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
#include <pwd.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <support/support.h>

#include "nss_test.h"

/* The specific values and names used here are arbitrary, other than
   correspondence (with suitable differences according to the tests as
   commented) between the given and expected entries.  */

static struct passwd pwd_table[] = {
  PWD (100),  /* baseline, matches */
  PWD (300),  /* wrong name and uid */
  PWD_N (200, NULL), /* missing name */
  PWD (60), /* unexpected name */
  { .pw_name = (char *)"name20000",  .pw_passwd = (char *) "*", .pw_uid = 20000,  \
    .pw_gid = 200, .pw_gecos = (char *) "*", .pw_dir = (char *) "*",	\
    .pw_shell = (char *) "*" }, /* wrong gid */
  { .pw_name = (char *)"name2",  .pw_passwd = (char *) "x", .pw_uid = 2,  \
    .pw_gid = 2, .pw_gecos = (char *) "y", .pw_dir = (char *) "z",	\
    .pw_shell = (char *) "*" }, /* spot check other text fields */
  PWD_LAST ()
};

static struct passwd exp_table[] = {
  PWD (100),
  PWD (30),
  PWD (200),
  PWD_N (60, NULL),
  PWD (20000),
  PWD (2),
  PWD_LAST ()
};

void
_nss_test1_init_hook(test_tables *t)
{
  t->pwd_table = pwd_table;
}

static int
do_test (void)
{
  int retval = 0;
  int i;
  struct passwd *p;

  __nss_configure_lookup ("passwd", "test1");

  setpwent ();

  i = 0;
  for (p = getpwent ();
       p != NULL && ! PWD_ISLAST (& exp_table[i]);
       ++i, p = getpwent ())
    retval += compare_passwds (i, p, & exp_table[i]);

  endpwent ();


  if (p)
    {
      printf ("FAIL: [?] passwd entry %u.%s unexpected\n", p->pw_uid, p->pw_name);
      ++retval;
    }
  if (! PWD_ISLAST (& exp_table[i]))
    {
      printf ("FAIL: [%d] passwd entry %u.%s missing\n", i,
	      exp_table[i].pw_uid, exp_table[i].pw_name);
      ++retval;
    }

#define EXPECTED 9
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
