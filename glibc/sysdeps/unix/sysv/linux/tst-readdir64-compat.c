/* Test readdir64 compatibility symbol.
   Copyright (C) 2018-2021 Free Software Foundation, Inc.
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

#include <dirent.h>
#include <dlfcn.h>
#include <errno.h>
#include <shlib-compat.h>
#include <stdbool.h>
#include <stdio.h>
#include <string.h>
#include <support/check.h>

/* Copied from <olddirent.h>.  */
struct __old_dirent64
  {
    __ino_t d_ino;
    __off64_t d_off;
    unsigned short int d_reclen;
    unsigned char d_type;
    char d_name[256];
  };

typedef struct __old_dirent64 *(*compat_readdir64_type) (DIR *);

struct __old_dirent64 *compat_readdir64 (DIR *);
compat_symbol_reference (libc, compat_readdir64, readdir64, GLIBC_2_1);

static int
do_test (void)
{
  /* Directory stream using the non-compat readdir64 symbol.  The test
     checks against this.  */
  DIR *dir_reference = opendir (".");
  TEST_VERIFY_EXIT (dir_reference != NULL);
  DIR *dir_test = opendir (".");
  TEST_VERIFY_EXIT (dir_test != NULL);

  /* This loop assumes that the enumeration order is consistent for
     two different handles.  Nothing should write to the current
     directory (in the source tree) while this test runs, so there
     should not be any difference due to races.  */
  size_t count = 0;
  while (true)
    {
      errno = 0;
      struct dirent64 *entry_reference = readdir64 (dir_reference);
      if (entry_reference == NULL && errno != 0)
        FAIL_EXIT1 ("readdir64 entry %zu: %m\n", count);
      struct __old_dirent64 *entry_test = compat_readdir64 (dir_test);
      if (entry_reference == NULL)
        {
          if (errno == EOVERFLOW)
            {
              TEST_VERIFY (entry_reference->d_ino
                           != (__ino_t) entry_reference->d_ino);
              printf ("info: inode number overflow at entry %zu\n", count);
              break;
            }
          if (errno != 0)
            FAIL_EXIT1 ("compat readdir64 entry %zu: %m\n", count);
        }

      /* Check that both streams end at the same time.  */
      if (entry_reference == NULL)
        {
          TEST_VERIFY (entry_test == NULL);
          break;
        }
      else
        TEST_VERIFY_EXIT (entry_test != NULL);

      /* d_off is never zero because it is the offset of the next
         entry (not the current entry).  */
      TEST_VERIFY (entry_reference->d_off > 0);

      /* Check that the entries are the same.  */
      TEST_COMPARE_BLOB (entry_reference->d_name,
                         strlen (entry_reference->d_name),
                         entry_test->d_name, strlen (entry_test->d_name));
      TEST_COMPARE (entry_reference->d_ino, entry_test->d_ino);
      TEST_COMPARE (entry_reference->d_off, entry_test->d_off);
      TEST_COMPARE (entry_reference->d_type, entry_test->d_type);
      TEST_COMPARE (entry_reference->d_reclen, entry_test->d_reclen);

      ++count;
    }
  printf ("info: %zu directory entries found\n", count);
  TEST_VERIFY (count >= 3);     /* ".", "..", and some source files.  */

  TEST_COMPARE (closedir (dir_test), 0);
  TEST_COMPARE (closedir (dir_reference), 0);
  return 0;
}

#include <support/test-driver.c>
