/* Test for fgetsgent_r and buffer sizes.
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

#include <array_length.h>
#include <errno.h>
#include <gshadow.h>
#include <stdbool.h>
#include <stdlib.h>
#include <support/check.h>
#include <support/support.h>
#include <support/temp_file.h>
#include <support/xmemstream.h>
#include <support/xstdio.h>

/* Turn a parsed struct back into a line string.  The returned string
   should be freed.  */
static char *
format_ent (const struct sgrp *e)
{
  struct xmemstream stream;
  xopen_memstream (&stream);
  TEST_COMPARE (putsgent (e, stream.out), 0);
  xfclose_memstream (&stream);
  return stream.buffer;
}

/* An entry in the input file along with the expected output.  */
struct input
{
  const char *line;		/* Line in the file.  */
  const char *expected;		/* Expected output.  NULL if skipped.  */
};

const struct input inputs[] =
  {
   /* Regular entries.  */
   { "g1:x1::\n", "g1:x1::\n" },
   { "g2:x2:a1:\n", "g2:x2:a1:\n" },
   { "g3:x3:a2:u1\n", "g3:x3:a2:u1\n" },
   { "g4:x4:a3,a4:u2,u3,u4\n", "g4:x4:a3,a4:u2,u3,u4\n" },

   /* Comments and empty lines.  */
   { "\n", NULL },
   { " \n", NULL },
   { "\t\n", NULL },
   { "#g:x::\n", NULL },
   { " #g:x::\n", NULL },
   { "\t#g:x::\n", NULL },
   { " \t#g:x::\n", NULL },

   /* Marker for synchronization.  */
   { "g5:x5::\n", "g5:x5::\n" },

   /* Leading whitespace.  */
   { " g6:x6::\n", "g6:x6::\n" },
   { "\tg7:x7::\n", "g7:x7::\n" },

   /* This is expected to trigger buffer exhaustion during parsing
      (bug 20338).  */
   {
    "g8:xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx:u5,u6,u7,u8,u9:\n",
    "g8:xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx:u5,u6,u7,u8,u9:\n",
   },
   {
    "g9:xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx::a5,a6,a7,a8,a9,a10\n",
    "g9:xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx::a5,a6,a7,a8,a9,a10\n",
   },
  };

/* Writes the test data to a temporary file and returns its name.  The
   returned pointer should be freed.  */
static char *
create_test_file (void)
{
  char *path;
  int fd = create_temp_file ("tst-fgetsgent_r-", &path);
  FILE *fp = fdopen (fd, "w");
  TEST_VERIFY_EXIT (fp != NULL);

  for (size_t i = 0; i < array_length (inputs); ++i)
    fputs (inputs[i].line, fp);

  xfclose (fp);
  return path;
}

/* Read the test file with the indicated start buffer size.  Return
   true if the buffer size had to be increased during reading.  */
static bool
run_test (const char *path, size_t buffer_size)
{
  bool resized = false;
  FILE *fp = xfopen (path, "r");

  /* This avoids repeated lseek system calls (bug 26257).  */
  TEST_COMPARE (fseeko64 (fp, 0, SEEK_SET), 0);

  size_t i = 0;
  while (true)
    {
      /* Skip over unused expected entries.  */
      while (i < array_length (inputs) && inputs[i].expected == NULL)
	++i;

      /* Store the data on the heap, to help valgrind to detect
	 invalid accesses.  */
      struct sgrp *result_storage = xmalloc (sizeof (*result_storage));
      char *buffer = xmalloc (buffer_size);
      struct sgrp **result_pointer_storage
	= xmalloc (sizeof (*result_pointer_storage));

      int ret = fgetsgent_r (fp, result_storage, buffer, buffer_size,
			     result_pointer_storage);
      if (ret == 0)
	{
	  TEST_VERIFY (*result_pointer_storage != NULL);
	  TEST_VERIFY (i < array_length (inputs));
	  if (*result_pointer_storage != NULL
	      && i < array_length (inputs))
	    {
	      char * actual = format_ent (*result_pointer_storage);
	      TEST_COMPARE_STRING (inputs[i].expected, actual);
	      free (actual);
	      ++i;
	    }
	  else
	    break;
	}
      else
	{
	  TEST_VERIFY (*result_pointer_storage == NULL);
	  TEST_COMPARE (ret, errno);

	  if (ret == ENOENT)
	    {
	      TEST_COMPARE (i, array_length (inputs));
	      free (result_pointer_storage);
	      free (buffer);
	      free (result_storage);
	      break;
	    }
	  else if (ret == ERANGE)
	    {
	      resized = true;
	      ++buffer_size;
	    }
	  else
	    FAIL_EXIT1 ("read failure: %m");
	}

      free (result_pointer_storage);
      free (buffer);
      free (result_storage);
    }

  xfclose (fp);
  return resized;
}

static int
do_test (void)
{
  char *path = create_test_file ();

  for (size_t buffer_size = 3; ; ++buffer_size)
    {
      bool resized = run_test (path, buffer_size);
      if (!resized)
	break;
    }

  free (path);

  return 0;
}

#include <support/test-driver.c>
