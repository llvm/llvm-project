/* Basic test of passwd database handling.
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

static int hook_called = 0;

/* Note: the values chosen here are arbitrary; they need only be
   unique within the table.  However, they do need to match the
   "pwdids" array further down.  */
static struct passwd pwd_table[] = {
    PWD (100),
    PWD (30),
    PWD (200),
    PWD (60),
    PWD (20000),
    PWD_LAST ()
  };

void
_nss_test1_init_hook(test_tables *t)
{
  hook_called = 1;
  t->pwd_table = pwd_table;
}

static int
do_test (void)
{
  int retval = 0;

  __nss_configure_lookup ("passwd", "test1");

  /* This must match the pwd_table above.  */
  static const unsigned int pwdids[] = { 100, 30, 200, 60, 20000 };
#define npwdids (sizeof (pwdids) / sizeof (pwdids[0]))

  setpwent ();

  const unsigned int *np = pwdids;
  for (struct passwd *p = getpwent (); p != NULL; ++np, p = getpwent ())
    {
      retval += compare_passwds (np-pwdids, p, & pwd_table[np-pwdids]);

      if (p->pw_uid != *np || strncmp (p->pw_name, "name", 4) != 0
	  || atol (p->pw_name + 4) != *np)
	{
	  printf ("FAIL: passwd entry %td wrong (%s, %u)\n",
		  np - pwdids, p->pw_name, p->pw_uid);
	  retval = 1;
	  break;
	}
    }

  endpwent ();

  for (int i = npwdids - 1; i >= 0; --i)
    {
      char buf[30];
      snprintf (buf, sizeof (buf), "name%u", pwdids[i]);

      struct passwd *p = getpwnam (buf);
      if (p == NULL || p->pw_uid != pwdids[i] || strcmp (buf, p->pw_name) != 0)
	{
	  printf ("FAIL: passwd entry \"%s\" wrong\n", buf);
	  retval = 1;
	}

      p = getpwuid (pwdids[i]);
      if (p == NULL || p->pw_uid != pwdids[i] || strcmp (buf, p->pw_name) != 0)
	{
	  printf ("FAIL: passwd entry %u wrong\n", pwdids[i]);
	  retval = 1;
	}

      snprintf (buf, sizeof (buf), "name%u", pwdids[i] + 1);

      p = getpwnam (buf);
      if (p != NULL)
	{
	  printf ("FAIL: passwd entry \"%s\" wrong\n", buf);
	  retval = 1;
	}

      p = getpwuid (pwdids[i] + 1);
      if (p != NULL)
	{
	  printf ("FAIL: passwd entry %u wrong\n", pwdids[i] + 1);
	  retval = 1;
	}
    }

  if (!hook_called)
    {
      retval = 1;
      printf("FAIL: init hook never called\n");
    }

  return retval;
}

#include <support/test-driver.c>
