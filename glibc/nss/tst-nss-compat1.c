/* Test error checking for group entries.
   Copyright (C) 2021 Free Software Foundation, Inc.
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

#include <shadow.h>

#include <support/support.h>
#include <support/check.h>

#include "nss_test.h"

static struct passwd pwd_table[] = {
    PWD (100),
    PWD (30),
    PWD_LAST ()
  };

static struct spwd spwd_table[] = {
    SPWD (100),
    SPWD (30),
    SPWD_LAST ()
  };

void
_nss_test1_init_hook(test_tables *t)
{
  t->pwd_table = pwd_table;
  t->spwd_table = spwd_table;
}

static int
do_test (void)
{
  struct passwd *p = NULL;
  struct spwd *s = NULL;
  struct group *g = NULL;

  /* Test that compat-to-test works.  */
  p = getpwuid (100);
  if (p == NULL)
    FAIL_EXIT1("getpwuid-compat-test1 p");
  else if (strcmp (p->pw_name, "name100") != 0)
    FAIL_EXIT1("getpwuid-compat-test1 name100");

  /* Shadow compat should use passwd via the alternate name.  */
  s = getspnam ("name30");
  if (s == NULL)
    FAIL_EXIT1("getspnam-compat-test1 s");
  else if (strcmp (s->sp_namp, "name30") != 0)
    FAIL_EXIT1("getpwuid-compat-test1 name30");

  /* Test that internal defconfig works.  */
  g = getgrgid (100);
  if (g == NULL)
    FAIL_EXIT1("getgrgid-compat-null");
  if (strcmp (g->gr_name, "wilma") != 0)
    FAIL_EXIT1("getgrgid-compat-name");

  return 0;
}

#include <support/test-driver.c>
