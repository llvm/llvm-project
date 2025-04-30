/* Test for reading directories with getdents64.
   Copyright (C) 2019-2021 Free Software Foundation, Inc.
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
#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <support/check.h>
#include <support/support.h>
#include <support/xunistd.h>
#include <sys/mman.h>
#include <unistd.h>

/* Called by large_buffer_checks below.  */
static void
large_buffer_check (int fd, char *large_buffer, size_t large_buffer_size)
{
  xlseek (fd, 0, SEEK_SET);
  ssize_t ret = getdents64 (fd, large_buffer, large_buffer_size);
  if (ret < 0)
    FAIL_EXIT1 ("getdents64 for buffer of %zu bytes failed: %m",
                large_buffer_size);
  if (ret < offsetof (struct dirent64, d_name))
    FAIL_EXIT1 ("getdents64 for buffer of %zu returned small value %zd",
                large_buffer_size, ret);
}

/* Bug 24740: Make sure that the system call argument is adjusted
   properly for the int type.  A large value should stay a large
   value, and not wrap around to something small, causing the system
   call to fail with EINVAL.  */
static void
large_buffer_checks (int fd)
{
  size_t large_buffer_size;
  if (!__builtin_add_overflow (UINT_MAX, 2, &large_buffer_size))
    {
      int flags = MAP_ANONYMOUS | MAP_PRIVATE;
#ifdef MAP_NORESERVE
      flags |= MAP_NORESERVE;
#endif
      void *large_buffer = mmap (NULL, large_buffer_size,
                                 PROT_READ | PROT_WRITE, flags, -1, 0);
      if (large_buffer == MAP_FAILED)
        printf ("warning: could not allocate %zu bytes of memory,"
                " subtests skipped\n", large_buffer_size);
      else
        {
          large_buffer_check (fd, large_buffer, INT_MAX);
          large_buffer_check (fd, large_buffer, (size_t) INT_MAX + 1);
          large_buffer_check (fd, large_buffer, (size_t) INT_MAX + 2);
          large_buffer_check (fd, large_buffer, UINT_MAX);
          large_buffer_check (fd, large_buffer, (size_t) UINT_MAX + 1);
          large_buffer_check (fd, large_buffer, (size_t) UINT_MAX + 2);
          xmunmap (large_buffer, large_buffer_size);
        }
    }
}

static void
do_test_large_size (void)
{
  int fd = xopen (".", O_RDONLY | O_DIRECTORY, 0);
  TEST_VERIFY (fd >= 0);
  large_buffer_checks (fd);

  xclose (fd);
}

static void
do_test_by_size (size_t buffer_size)
{
  /* The test compares the iteration order with readdir64.  */
  DIR *reference = opendir (".");
  TEST_VERIFY_EXIT (reference != NULL);

  int fd = xopen (".", O_RDONLY | O_DIRECTORY, 0);
  TEST_VERIFY (fd >= 0);

  /* Perform two passes, with a rewind operating between passes.  */
  for (int pass = 0; pass < 2; ++pass)
    {
      /* Check that we need to fill the buffer multiple times.  */
      int read_count = 0;

      while (true)
        {
          /* Simple way to make sure that the memcpy below does not read
             non-existing data.  */
          struct
          {
            char buffer[buffer_size];
            struct dirent64 pad;
          } data;

          ssize_t ret = getdents64 (fd, &data.buffer, sizeof (data.buffer));
          if (ret < 0)
            FAIL_EXIT1 ("getdents64: %m");
          if (ret == 0)
            break;
          ++read_count;

          char *current = data.buffer;
          char *end = data.buffer + ret;
          while (current != end)
            {
              struct dirent64 entry;
              memcpy (&entry, current, sizeof (entry));
              /* Truncate overlong strings.  */
              entry.d_name[sizeof (entry.d_name) - 1] = '\0';
              TEST_VERIFY (strlen (entry.d_name) < sizeof (entry.d_name) - 1);

              errno = 0;
              struct dirent64 *refentry = readdir64 (reference);
              if (refentry == NULL && errno == 0)
                FAIL_EXIT1 ("readdir64 failed too early, at: %s",
                            entry.d_name);
              else if (refentry == NULL)
                FAIL_EXIT1 ("readdir64: %m");

              TEST_COMPARE_STRING (entry.d_name, refentry->d_name);
              TEST_COMPARE (entry.d_ino, refentry->d_ino);
              TEST_COMPARE (entry.d_off, refentry->d_off);
              TEST_COMPARE (entry.d_type, refentry->d_type);

              /* Offset zero is reserved for the first entry.  */
              TEST_VERIFY (entry.d_off != 0);

              TEST_VERIFY_EXIT (entry.d_reclen <= end - current);
              current += entry.d_reclen;
            }
        }

      /* We expect to have reached the end of the stream.  */
      errno = 0;
      TEST_VERIFY (readdir64 (reference) == NULL);
      TEST_COMPARE (errno, 0);

      /* direntries_read has been called more than once.  */
      TEST_VERIFY (read_count > 0);

      /* Rewind both directory streams.  */
      xlseek (fd, 0, SEEK_SET);
      rewinddir (reference);
    }

  xclose (fd);
  closedir (reference);
}

static int
do_test (void)
{
  do_test_by_size (512);
  do_test_by_size (1024);
  do_test_by_size (4096);

  do_test_large_size ();

  return 0;
}

#include <support/test-driver.c>
