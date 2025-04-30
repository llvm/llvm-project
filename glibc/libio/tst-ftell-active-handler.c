/* Verify that ftell returns the correct value at various points before and
   after the handler on which it is called becomes active.
   Copyright (C) 2014-2021 Free Software Foundation, Inc.
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

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <unistd.h>
#include <fcntl.h>
#include <locale.h>
#include <wchar.h>

static int do_test (void);

#define TEST_FUNCTION do_test ()
#include "../test-skeleton.c"

#define get_handles_fdopen(filename, fd, fp, fd_mode, mode) \
({									      \
  int ret = 0;								      \
  (fd) = open ((filename), (fd_mode), 0);				      \
  if ((fd) == -1)							      \
    {									      \
      printf ("open failed: %m\n");					      \
      ret = 1;								      \
    }									      \
  else									      \
    {									      \
      (fp) = fdopen ((fd), (mode));					      \
      if ((fp) == NULL)							      \
	{								      \
	  printf ("fdopen failed: %m\n");				      \
	  close (fd);							      \
	  ret = 1;							      \
	}								      \
    }									      \
  ret;									      \
})

#define get_handles_fopen(filename, fd, fp, mode) \
({									      \
  int ret = 0;								      \
  (fp) = fopen ((filename), (mode));					      \
  if ((fp) == NULL)							      \
    {									      \
      printf ("fopen failed: %m\n");					      \
      ret = 1;								      \
    }									      \
  else									      \
    {									      \
      (fd) = fileno (fp);						      \
      if ((fd) == -1)							      \
	{								      \
	  printf ("fileno failed: %m\n");				      \
	  ret = 1;							      \
	}								      \
    }									      \
  ret;									      \
})

/* data points to either char_data or wide_data, depending on whether we're
   testing regular file mode or wide mode respectively.  Similarly,
   fputs_func points to either fputs or fputws.  data_len keeps track of the
   length of the current data and file_len maintains the current file
   length.  */
static const void *data;
static const char *char_data = "abcdef";
static const wchar_t *wide_data = L"abcdef";
static size_t data_len;
static size_t file_len;

typedef int (*fputs_func_t) (const void *data, FILE *fp);
typedef void *(*fgets_func_t) (void *ws, int n, FILE *fp);
fputs_func_t fputs_func;
fgets_func_t fgets_func;

/* This test verifies that the offset reported by ftell is correct after the
   file is truncated using ftruncate.  ftruncate does not change the file
   offset on truncation and hence, SEEK_CUR should continue to point to the
   old offset and not be changed to the new offset.  */
static int
do_ftruncate_test (const char *filename)
{
  FILE *fp = NULL;
  int fd;
  int ret = 0;
  struct test
    {
      const char *mode;
      int fd_mode;
    } test_modes[] = {
	  {"r+", O_RDWR},
	  {"w", O_WRONLY | O_TRUNC},
	  {"w+", O_RDWR | O_TRUNC},
	  {"a", O_WRONLY},
	  {"a+", O_RDWR}
    };

  for (int j = 0; j < 2; j++)
    {
      for (int i = 0; i < sizeof (test_modes) / sizeof (struct test); i++)
	{
	  int fileret;
	  printf ("\tftruncate: %s (file, \"%s\"): ",
		  j == 0 ? "fopen" : "fdopen",
		  test_modes[i].mode);

	  if (j == 0)
	    fileret = get_handles_fopen (filename, fd, fp, test_modes[i].mode);
	  else
	    fileret = get_handles_fdopen (filename, fd, fp,
					  test_modes[i].fd_mode,
					  test_modes[i].mode);

	  if (fileret != 0)
	    return fileret;

	  /* Write some data.  */
	  size_t written = fputs_func (data, fp);

	  if (written == EOF)
	    {
	      printf ("fputs[1] failed to write data\n");
	      ret |= 1;
	    }

	  /* Record the offset.  */
	  long offset = ftell (fp);

	  /* Flush data to allow switching active handles.  */
	  if (fflush (fp))
	    {
	      printf ("Flush failed: %m\n");
	      ret |= 1;
	    }

	  /* Now truncate the file.  */
	  if (ftruncate (fd, 0) != 0)
	    {
	      printf ("Failed to truncate file: %m\n");
	      ret |= 1;
	    }

	  /* ftruncate does not change the offset, so there is no need to call
	     anything to be able to switch active handles.  */
	  long new_offset = ftell (fp);

	  /* The offset should remain unchanged since ftruncate does not update
	     it.  */
	  if (offset != new_offset)
	    {
	      printf ("Incorrect offset.  Expected %ld, but got %ld\n",
		      offset, new_offset);

	      ret |= 1;
	    }
	  else
	    printf ("offset = %ld\n", offset);

	  fclose (fp);
	}
    }

  return ret;
}
/* Test that ftell output after a rewind is correct.  */
static int
do_rewind_test (const char *filename)
{
  int ret = 0;
  struct test
    {
      const char *mode;
      int fd_mode;
      size_t old_off;
      size_t new_off;
    } test_modes[] = {
	  {"w", O_WRONLY | O_TRUNC, 0, data_len},
	  {"w+", O_RDWR | O_TRUNC, 0, data_len},
	  {"r+", O_RDWR, 0, data_len},
	  /* The new offsets for 'a' and 'a+' modes have to factor in the
	     previous writes since they always append to the end of the
	     file.  */
	  {"a", O_WRONLY, 0, 3 * data_len},
	  {"a+", O_RDWR, 0, 4 * data_len},
    };

  /* Empty the file before the test so that our offsets are simple to
     calculate.  */
  FILE *fp = fopen (filename, "w");
  if (fp == NULL)
    {
      printf ("Failed to open file for emptying\n");
      return 1;
    }
  fclose (fp);

  for (int j = 0; j < 2; j++)
    {
      for (int i = 0; i < sizeof (test_modes) / sizeof (struct test); i++)
	{
	  FILE *fp;
	  int fd;
	  int fileret;

	  printf ("\trewind: %s (file, \"%s\"): ", j == 0 ? "fdopen" : "fopen",
		  test_modes[i].mode);

	  if (j == 0)
	    fileret = get_handles_fdopen (filename, fd, fp,
					  test_modes[i].fd_mode,
					  test_modes[i].mode);
	  else
	    fileret = get_handles_fopen (filename, fd, fp, test_modes[i].mode);

	  if (fileret != 0)
	    return fileret;

	  /* Write some content to the file, rewind and ensure that the ftell
	     output after the rewind is 0.  POSIX does not specify what the
	     behavior is when a file is rewound in 'a' mode, so we retain
	     current behavior, which is to keep the 0 offset.  */
	  size_t written = fputs_func (data, fp);

	  if (written == EOF)
	    {
	      printf ("fputs[1] failed to write data\n");
	      ret |= 1;
	    }

	  rewind (fp);
	  long offset = ftell (fp);

	  if (offset != test_modes[i].old_off)
	    {
	      printf ("Incorrect old offset.  Expected %zu, but got %ld, ",
		      test_modes[i].old_off, offset);
	      ret |= 1;
	    }
	  else
	    printf ("old offset = %ld, ", offset);

	  written = fputs_func (data, fp);

	  if (written == EOF)
	    {
	      printf ("fputs[1] failed to write data\n");
	      ret |= 1;
	    }

	  /* After this write, the offset in append modes should factor in the
	     implicit lseek to the end of file.  */
	  offset = ftell (fp);
	  if (offset != test_modes[i].new_off)
	    {
	      printf ("Incorrect new offset.  Expected %zu, but got %ld\n",
		      test_modes[i].new_off, offset);
	      ret |= 1;
	    }
	  else
	    printf ("new offset = %ld\n", offset);
	}
    }
  return ret;
}

/* Test that the value of ftell is not cached when the stream handle is not
   active.  */
static int
do_ftell_test (const char *filename)
{
  int ret = 0;
  struct test
    {
      const char *mode;
      int fd_mode;
      size_t old_off;
      size_t new_off;
      size_t eof_off;
    } test_modes[] = {
	  /* In w, w+ and r+ modes, the file position should be at the
	     beginning of the file.  After the write, the offset should be
	     updated to data_len.  We don't use eof_off in w and a modes since
	     they don't allow reading.  */
	  {"w", O_WRONLY | O_TRUNC, 0, data_len, 0},
	  {"w+", O_RDWR | O_TRUNC, 0, data_len, 2 * data_len},
	  {"r+", O_RDWR, 0, data_len, 3 * data_len},
	  /* For the 'a' mode, the initial file position should be the
	     current end of file. After the write, the offset has data_len
	     added to the old value.  For a+ mode however, the initial file
	     position is the file position of the underlying file descriptor,
	     since it is initially assumed to be in read mode.  */
	  {"a", O_WRONLY, 3 * data_len, 4 * data_len, 5 * data_len},
	  {"a+", O_RDWR, 0, 5 * data_len, 6 * data_len},
    };
  for (int j = 0; j < 2; j++)
    {
      for (int i = 0; i < sizeof (test_modes) / sizeof (struct test); i++)
	{
	  FILE *fp;
	  int fd;
	  int fileret;

	  printf ("\tftell: %s (file, \"%s\"): ", j == 0 ? "fdopen" : "fopen",
		  test_modes[i].mode);

	  if (j == 0)
	    fileret = get_handles_fdopen (filename, fd, fp,
					  test_modes[i].fd_mode,
					  test_modes[i].mode);
	  else
	    fileret = get_handles_fopen (filename, fd, fp, test_modes[i].mode);

	  if (fileret != 0)
	    return fileret;

	  long off = ftell (fp);
	  if (off != test_modes[i].old_off)
	    {
	      printf ("Incorrect old offset.  Expected %zu but got %ld, ",
		      test_modes[i].old_off, off);
	      ret |= 1;
	    }
	  else
	    printf ("old offset = %ld, ", off);

	  /* The effect of this write on the offset should be seen in the ftell
	     call that follows it.  */
	  int write_ret = write (fd, data, data_len);
	  if (write_ret != data_len)
	    {
	      printf ("write failed (%m)\n");
	      ret |= 1;
	    }
	  off = ftell (fp);

	  if (off != test_modes[i].new_off)
	    {
	      printf ("Incorrect new offset.  Expected %zu but got %ld",
		      test_modes[i].new_off, off);
	      ret |= 1;
	    }
	  else
	    printf ("new offset = %ld", off);

	  /* Read to the end, write some data to the fd and check if ftell can
	     see the new ofset.  Do this test only for files that allow
	     reading.  */
	  if (test_modes[i].fd_mode != O_WRONLY)
	    {
	      wchar_t tmpbuf[data_len];

	      rewind (fp);

	      while (fgets_func (tmpbuf, data_len, fp) && !feof (fp));

	      write_ret = write (fd, data, data_len);
	      if (write_ret != data_len)
		{
		  printf ("write failed (%m)\n");
		  ret |= 1;
		}
	      off = ftell (fp);

	      if (off != test_modes[i].eof_off)
		{
		  printf (", Incorrect offset after read EOF.  "
			  "Expected %zu but got %ld\n",
			  test_modes[i].eof_off, off);
		  ret |= 1;
		}
	      else
		printf (", offset after EOF = %ld\n", off);
	    }
	  else
	    putc ('\n', stdout);

	  fclose (fp);
	}
    }

  return ret;
}

/* This test opens the file for writing, moves the file offset of the
   underlying file, writes out data and then checks if ftell trips on it.  */
static int
do_write_test (const char *filename)
{
  FILE *fp = NULL;
  int fd;
  int ret = 0;
  struct test
    {
      const char *mode;
      int fd_mode;
    } test_modes[] = {
	  {"w", O_WRONLY | O_TRUNC},
	  {"w+", O_RDWR | O_TRUNC},
	  {"r+", O_RDWR}
    };

  for (int j = 0; j < 2; j++)
    {
      for (int i = 0; i < sizeof (test_modes) / sizeof (struct test); i++)
	{
	  int fileret;
	  printf ("\twrite: %s (file, \"%s\"): ", j == 0 ? "fopen" : "fdopen",
		  test_modes[i].mode);

	  if (j == 0)
	    fileret = get_handles_fopen (filename, fd, fp, test_modes[i].mode);
	  else
	    fileret = get_handles_fdopen (filename, fd, fp,
					  test_modes[i].fd_mode,
					  test_modes[i].mode);

	  if (fileret != 0)
	    return fileret;

	  /* Move offset to just before the end of the file.  */
	  off_t seek_ret = lseek (fd, file_len - 1, SEEK_SET);
	  if (seek_ret == -1)
	    {
	      printf ("lseek failed: %m\n");
	      ret |= 1;
	    }

	  /* Write some data.  */
	  size_t written = fputs_func (data, fp);

	  if (written == EOF)
	    {
	      printf ("fputs[1] failed to write data\n");
	      ret |= 1;
	    }

	  /* Verify that the offset points to the end of the file.  The length
	     of the file would be the original length + the length of data
	     written to it - the amount by which we moved the offset using
	     lseek.  */
	  long offset = ftell (fp);
	  file_len = file_len - 1 + data_len;

	  if (offset != file_len)
	    {
	      printf ("Incorrect offset.  Expected %zu, but got %ld\n",
		      file_len, offset);

	      ret |= 1;
	    }

	  printf ("offset = %ld\n", offset);
	  fclose (fp);
	}
    }

  return ret;
}

/* This test opens a file in append mode, writes some data, and then verifies
   that ftell does not trip over it.  */
static int
do_append_test (const char *filename)
{
  FILE *fp = NULL;
  int ret = 0;
  int fd;

  struct test
    {
      const char *mode;
      int fd_mode;
    } test_modes[] = {
	  {"a", O_WRONLY},
	  {"a+", O_RDWR}
    };

  for (int j = 0; j < 2; j++)
    {
      for (int i = 0; i < sizeof (test_modes) / sizeof (struct test); i++)
	{
	  int fileret;

	  printf ("\tappend: %s (file, \"%s\"): ", j == 0 ? "fopen" : "fdopen",
		  test_modes[i].mode);

	  if (j == 0)
	    fileret = get_handles_fopen (filename, fd, fp, test_modes[i].mode);
	  else
	    fileret = get_handles_fdopen (filename, fd, fp,
					  test_modes[i].fd_mode,
					  test_modes[i].mode);

	  if (fileret != 0)
	    return fileret;

	  /* Write some data.  */
	  size_t written = fputs_func (data, fp);

	  if (written == EOF)
	    {
	      printf ("fputs[1] failed to write all data\n");
	      ret |= 1;
	    }

	  /* Verify that the offset points to the end of the file.  The file
	     len is maintained by adding data_len each time to reflect the data
	     written to it.  */
	  long offset = ftell (fp);
	  file_len += data_len;

	  if (offset != file_len)
	    {
	      printf ("Incorrect offset.  Expected %zu, but got %ld\n",
		      file_len, offset);

	      ret |= 1;
	    }

	  printf ("offset = %ld\n", offset);
	  fclose (fp);
	}
    }

  /* For fdopen in 'a' mode, the file descriptor should not change if the file
     is already open with the O_APPEND flag set.  */
  fd = open (filename, O_WRONLY | O_APPEND, 0);
  if (fd == -1)
    {
      printf ("open(O_APPEND) failed: %m\n");
      return 1;
    }

  off_t seek_ret = lseek (fd, file_len - 1, SEEK_SET);
  if (seek_ret == -1)
    {
      printf ("lseek[O_APPEND][0] failed: %m\n");
      ret |= 1;
    }

  fp = fdopen (fd, "a");
  if (fp == NULL)
    {
      printf ("fdopen(O_APPEND) failed: %m\n");
      close (fd);
      return 1;
    }

  off_t new_seek_ret = lseek (fd, 0, SEEK_CUR);
  if (seek_ret == -1)
    {
      printf ("lseek[O_APPEND][1] failed: %m\n");
      ret |= 1;
    }

  printf ("\tappend: fdopen (file, \"a\"): O_APPEND: ");

  if (seek_ret != new_seek_ret)
    {
      printf ("incorrectly modified file offset to %jd, should be %jd",
	      (intmax_t)  new_seek_ret, (intmax_t) seek_ret);
      ret |= 1;
    }
  else
    printf ("retained current file offset %jd", (intmax_t) seek_ret);

  new_seek_ret = ftello (fp);

  if (seek_ret != new_seek_ret)
    {
      printf (", ftello reported incorrect offset %jd, should be %jd\n",
	      (intmax_t) new_seek_ret, (intmax_t) seek_ret);
      ret |= 1;
    }
  else
    printf (", ftello reported correct offset %jd\n", (intmax_t) seek_ret);

  fclose (fp);

  return ret;
}

static int
do_one_test (const char *filename)
{
  int ret = 0;

  ret |= do_ftell_test (filename);
  ret |= do_write_test (filename);
  ret |= do_append_test (filename);
  ret |= do_rewind_test (filename);
  ret |= do_ftruncate_test (filename);

  return ret;
}

/* Run a set of tests for ftell for regular files and wide mode files.  */
static int
do_test (void)
{
  int ret = 0;
  FILE *fp = NULL;
  char *filename;
  size_t written;
  int fd = create_temp_file ("tst-active-handler-tmp.", &filename);

  if (fd == -1)
    {
      printf ("create_temp_file: %m\n");
      return 1;
    }

  fp = fdopen (fd, "w");
  if (fp == NULL)
    {
      printf ("fdopen[0]: %m\n");
      close (fd);
      return 1;
    }

  data = char_data;
  data_len = strlen (char_data);
  file_len = strlen (char_data);
  written = fputs (data, fp);

  if (written == EOF)
    {
      printf ("fputs[1] failed to write data\n");
      ret = 1;
    }

  fclose (fp);
  if (ret)
    return ret;

  /* Tests for regular files.  */
  puts ("Regular mode:");
  fputs_func = (fputs_func_t) fputs;
  fgets_func = (fgets_func_t) fgets;
  data = char_data;
  data_len = strlen (char_data);
  ret |= do_one_test (filename);

  /* Truncate the file before repeating the tests in wide mode.  */
  fp = fopen (filename, "w");
  if (fp == NULL)
    {
      printf ("fopen failed %m\n");
      return 1;
    }
  fclose (fp);

  /* Tests for wide files.  */
  puts ("Wide mode:");
  if (setlocale (LC_ALL, "en_US.UTF-8") == NULL)
    {
      printf ("Cannot set en_US.UTF-8 locale.\n");
      return 1;
    }
  fputs_func = (fputs_func_t) fputws;
  fgets_func = (fgets_func_t) fgetws;
  data = wide_data;
  data_len = wcslen (wide_data);
  ret |= do_one_test (filename);

  return ret;
}
