/* Test the xreadlink function.
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

#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <support/check.h>
#include <support/support.h>
#include <support/temp_file.h>
#include <support/xunistd.h>

static int
do_test (void)
{
  char *dir = support_create_temp_directory ("tst-xreadlink-");
  char *symlink_name = xasprintf ("%s/symlink", dir);
  add_temp_file (symlink_name);

  /* The limit 10000 is arbitrary and simply there to prevent an
     attempt to exhaust all available disk space.  */
  for (int size = 1; size < 10000; ++size)
    {
      char *contents = xmalloc (size + 1);
      for (int i = 0; i < size; ++i)
        contents[i] = 'a' + (rand () % 26);
      contents[size] = '\0';
      if (symlink (contents, symlink_name) != 0)
        {
          if (errno == ENAMETOOLONG)
            {
              printf ("info: ENAMETOOLONG failure at %d bytes\n", size);
              free (contents);
              break;
            }
          FAIL_EXIT1 ("symlink (%d bytes): %m", size);
        }

      char *readlink_result = xreadlink (symlink_name);
      TEST_VERIFY (strcmp (readlink_result, contents) == 0);
      free (readlink_result);
      xunlink (symlink_name);
      free (contents);
    }

  /* Create an empty file to suppress the temporary file deletion
     warning.  */
  xclose (xopen (symlink_name, O_WRONLY | O_CREAT, 0));

  free (symlink_name);
  free (dir);

  return 0;
}

#include <support/test-driver.c>
