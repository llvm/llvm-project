/* Check for GLOB_TIDLE heap allocation issues (bugs 22320, 22325, 22332).
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

#include <glob.h>
#include <mcheck.h>
#include <nss.h>
#include <pwd.h>
#include <stdlib.h>
#include <string.h>
#include <support/check.h>
#include <support/support.h>

/* Flag which indicates whether to pass the GLOB_ONLYDIR flag.  */
static int do_onlydir;

/* Flag which indicates whether to pass the GLOB_NOCHECK flag.  */
static int do_nocheck;

/* Flag which indicates whether to pass the GLOB_MARK flag.  */
static int do_mark;

/* Flag which indicates whether to pass the GLOB_NOESCAPE flag.  */
static int do_noescape;

static void
one_test (const char *prefix, const char *middle, const char *suffix)
{
  char *pattern = xasprintf ("%s%s%s", prefix, middle, suffix);
  int flags = GLOB_TILDE;
  if (do_onlydir)
    flags |= GLOB_ONLYDIR;
  if (do_nocheck)
    flags |= GLOB_NOCHECK;
  if (do_mark)
    flags |= GLOB_MARK;
  if (do_noescape)
    flags |= GLOB_NOESCAPE;
  glob_t gl;
  /* This glob call might result in crashes or memory leaks.  */
  if (glob (pattern, flags, NULL, &gl) == 0)
    globfree (&gl);
  free (pattern);
}

enum
  {
    /* The largest base being tested.  */
    largest_base_size = 500000,

    /* The actual size is the base size plus a variable whose absolute
       value is not greater than this.  This helps malloc to trigger
       overflows.  */
    max_size_skew = 16,

    /* The maximum string length supported by repeating_string
       below.  */
    repeat_size = largest_base_size + max_size_skew,
  };

/* Used to construct strings which repeat a single character 'x'.  */
static char *repeat;

/* Return a string of SIZE characters.  */
const char *
repeating_string (int size)
{
  TEST_VERIFY (size >= 0);
  TEST_VERIFY (size <= repeat_size);
  const char *repeated_shifted = repeat + repeat_size - size;
  TEST_VERIFY (strlen (repeated_shifted) == size);
  return repeated_shifted;
}

static int
do_test (void)
{
  /* Avoid network-based NSS modules and initialize nss_files with a
     dummy lookup.  This has to come before mtrace because NSS does
     not free all memory.  */
  __nss_configure_lookup ("passwd", "files");
  (void) getpwnam ("root");

  mtrace ();

  repeat = xmalloc (repeat_size + 1);
  memset (repeat, 'x', repeat_size);
  repeat[repeat_size] = '\0';

  /* These numbers control the size of the user name.  The values
     cover the minimum (0), a typical size (8), a large
     stack-allocated size (100000), and a somewhat large
     heap-allocated size (largest_base_size).  */
  static const int base_sizes[] = { 0, 8, 100, 100000, largest_base_size, -1 };

  for (do_onlydir = 0; do_onlydir < 2; ++do_onlydir)
    for (do_nocheck = 0; do_nocheck < 2; ++do_nocheck)
      for (do_mark = 0; do_mark < 2; ++do_mark)
	for (do_noescape = 0; do_noescape < 2; ++do_noescape)
	  for (int base_idx = 0; base_sizes[base_idx] >= 0; ++base_idx)
	    {
	      for (int size_skew = -max_size_skew; size_skew <= max_size_skew;
		   ++size_skew)
		{
		  int size = base_sizes[base_idx] + size_skew;
		  if (size < 0)
		    continue;

		  const char *user_name = repeating_string (size);
		  one_test ("~", user_name, "/a/b");
		  one_test ("~", user_name, "x\\x\\x////x\\a");
		}

	      const char *user_name = repeating_string (base_sizes[base_idx]);
	      one_test ("~", user_name, "");
	      one_test ("~", user_name, "/");
	      one_test ("~", user_name, "/a");
	      one_test ("~", user_name, "/*/*");
	      one_test ("~", user_name, "\\/");
	      one_test ("/~", user_name, "");
	      one_test ("*/~", user_name, "/a/b");
	    }

  free (repeat);

  return 0;
}

#define TIMEOUT 200
#include <support/test-driver.c>
