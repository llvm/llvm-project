/* Basic test for two passwd databases.
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

/* The data in these tables is arbitrary, but the merged data based on
   the first two tables will be compared against the expected data in
   the pwd_expected table, and the tests[] array.  */

static struct passwd pwd_table_1[] = {
    PWD (100),
    PWD (30),
    PWD (200),
    PWD (60),
    PWD (20000),
    PWD_LAST ()
  };

static struct passwd pwd_table_2[] = {
    PWD (5),
    PWD_N(200, "name30"),
    PWD (16),
    PWD_LAST ()
  };

void
_nss_test1_init_hook(test_tables *t)
{
  t->pwd_table = pwd_table_1;
}

void
_nss_test2_init_hook(test_tables *t)
{
  t->pwd_table = pwd_table_2;
}

static struct passwd pwd_expected[] = {
  PWD(100),
  PWD(30),
  PWD(200),
  PWD(60),
  PWD(20000),
  PWD(5),
  PWD_N(200, "name30"),
  PWD(16),
  PWD_LAST ()
};

static struct {
  uid_t uid;
  const char *name;
} tests[] = {
  { 100, "name100" }, /* control, first db */
  {  16, "name16"  }, /* second db */
  {  30, "name30"  }, /* test overlaps in name */
  { 200, "name200" }, /* test overlaps uid */
  { 0, NULL }
};

static int
do_test (void)
{
  int retval = 0;
  int i;

  __nss_configure_lookup ("passwd", "test1 test2");

  setpwent ();

  i = 0;
  for (struct passwd *p = getpwent (); p != NULL; ++i, p = getpwent ())
    {
      retval += compare_passwds (i, & pwd_expected[i], p);

      if (p->pw_uid != pwd_expected[i].pw_uid || strcmp (p->pw_name, pwd_expected[i].pw_name) != 0)
      {
	printf ("FAIL: getpwent for %u.%s returned %u.%s\n",
		pwd_expected[i].pw_uid, pwd_expected[i].pw_name,
		p->pw_uid, p->pw_name);
	retval = 1;
	break;
      }
    }

  endpwent ();

  for (i=0; tests[i].name; i++)
    {
      struct passwd *p = getpwnam (tests[i].name);
      if (strcmp (p->pw_name, tests[i].name) != 0
	  || p->pw_uid != tests[i].uid)
	{
	  printf("FAIL: getpwnam for %u.%s returned %u.%s\n",
		 tests[i].uid, tests[i].name,
		 p->pw_uid, p->pw_name);
	  retval = 1;
	}

      p = getpwuid (tests[i].uid);
      if (strcmp (p->pw_name, tests[i].name) != 0
	  || p->pw_uid != tests[i].uid)
	{
	  printf("FAIL: getpwuid for %u.%s returned %u.%s\n",
		 tests[i].uid, tests[i].name,
		 p->pw_uid, p->pw_name);
	  retval = 1;
	}
    }

  return retval;
}

#include <support/test-driver.c>
