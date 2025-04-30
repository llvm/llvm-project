/* Copyright (C) 2015-2021 Free Software Foundation, Inc.
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
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <support/support.h>

int
do_test (void)
{
  __nss_configure_lookup ("passwd", "files");

  /* Count the number of entries in the password database, and fetch
     data from the first and last entries.  */
  size_t count = 0;
  struct passwd * pw;
  char *first_name = NULL;
  uid_t first_uid = 0;
  char *last_name = NULL;
  uid_t last_uid = 0;
  setpwent ();
  while ((pw  = getpwent ()) != NULL)
    {
      if (first_name == NULL)
	{
	  first_name = xstrdup (pw->pw_name);
	  first_uid = pw->pw_uid;
	}

      free (last_name);
      last_name = xstrdup (pw->pw_name);
      last_uid = pw->pw_uid;
      ++count;
    }
  endpwent ();

  if (count == 0)
    {
      printf ("No entries in the password database.\n");
      return 0;
    }

  /* Try again, this time interleaving with name-based and UID-based
     lookup operations.  The counts do not match if the interleaved
     lookups affected the enumeration.  */
  size_t new_count = 0;
  setpwent ();
  while ((pw  = getpwent ()) != NULL)
    {
      if (new_count == count)
	{
	  printf ("Additional entry in the password database.\n");
	  return 1;
	}
      ++new_count;
      struct passwd *pw2 = getpwnam (first_name);
      if (pw2 == NULL)
	{
	  printf ("getpwnam (%s) failed: %m\n", first_name);
	  return 1;
	}
      pw2 = getpwnam (last_name);
      if (pw2 == NULL)
	{
	  printf ("getpwnam (%s) failed: %m\n", last_name);
	  return 1;
	}
      pw2 = getpwuid (first_uid);
      if (pw2 == NULL)
	{
	  printf ("getpwuid (%llu) failed: %m\n",
		  (unsigned long long) first_uid);
	  return 1;
	}
      pw2 = getpwuid (last_uid);
      if (pw2 == NULL)
	{
	  printf ("getpwuid (%llu) failed: %m\n",
		  (unsigned long long) last_uid);
	  return 1;
	}
    }
  endpwent ();
  if (new_count < count)
    {
      printf ("Missing entry in the password database.\n");
      return 1;
    }

  return 0;
}

#define TIMEOUT 300
#include <support/test-driver.c>
