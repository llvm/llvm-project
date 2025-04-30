/* Tests of fseek and fseeko.
   Copyright (C) 2000-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Ulrich Drepper <drepper@redhat.com>, 2000.

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

#include <error.h>
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <time.h>
#include <sys/stat.h>


static int
do_test (void)
{
  const char *tmpdir;
  char *fname;
  int fd;
  FILE *fp;
  const char outstr[] = "hello world!\n";
  char strbuf[sizeof outstr];
  char buf[200];
  struct stat64 st1;
  struct stat64 st2;
  int result = 0;

  tmpdir = getenv ("TMPDIR");
  if (tmpdir == NULL || tmpdir[0] == '\0')
    tmpdir = "/tmp";

  asprintf (&fname, "%s/tst-fseek.XXXXXX", tmpdir);
  if (fname == NULL)
    error (EXIT_FAILURE, errno, "cannot generate name for temporary file");

  /* Create a temporary file.   */
  fd = mkstemp (fname);
  if (fd == -1)
    error (EXIT_FAILURE, errno, "cannot open temporary file");

  fp = fdopen (fd, "w+");
  if (fp == NULL)
    error (EXIT_FAILURE, errno, "cannot get FILE for temporary file");

  setbuffer (fp, strbuf, sizeof (outstr) -1);

  if (fwrite (outstr, sizeof (outstr) - 1, 1, fp) != 1)
    {
      printf ("%d: write error\n", __LINE__);
      result = 1;
      goto out;
    }

  /* The EOF flag must be reset.  */
  if (fgetc (fp) != EOF)
    {
      printf ("%d: managed to read at end of file\n", __LINE__);
      result = 1;
    }
  else if (! feof (fp))
    {
      printf ("%d: EOF flag not set\n", __LINE__);
      result = 1;
    }
  if (fseek (fp, 0, SEEK_CUR) != 0)
    {
      printf ("%d: fseek(fp, 0, SEEK_CUR) failed\n", __LINE__);
      result = 1;
    }
  else if (feof (fp))
    {
      printf ("%d: fseek() didn't reset EOF flag\n", __LINE__);
      result = 1;
    }

  /* Do the same for fseeko().  */
    if (fgetc (fp) != EOF)
    {
      printf ("%d: managed to read at end of file\n", __LINE__);
      result = 1;
    }
  else if (! feof (fp))
    {
      printf ("%d: EOF flag not set\n", __LINE__);
      result = 1;
    }
  if (fseeko (fp, 0, SEEK_CUR) != 0)
    {
      printf ("%d: fseek(fp, 0, SEEK_CUR) failed\n", __LINE__);
      result = 1;
    }
  else if (feof (fp))
    {
      printf ("%d: fseek() didn't reset EOF flag\n", __LINE__);
      result = 1;
    }

  /* Go back to the beginning of the file: absolute.  */
  if (fseek (fp, 0, SEEK_SET) != 0)
    {
      printf ("%d: fseek(fp, 0, SEEK_SET) failed\n", __LINE__);
      result = 1;
    }
  else if (fflush (fp) != 0)
    {
      printf ("%d: fflush() failed\n", __LINE__);
      result = 1;
    }
  else if (lseek (fd, 0, SEEK_CUR) != 0)
    {
      printf ("%d: lseek() returned different position\n", __LINE__);
      result = 1;
    }
  else if (fread (buf, sizeof (outstr) - 1, 1, fp) != 1)
    {
      printf ("%d: fread() failed\n", __LINE__);
      result = 1;
    }
  else if (memcmp (buf, outstr, sizeof (outstr) - 1) != 0)
    {
      printf ("%d: content after fseek(,,SEEK_SET) wrong\n", __LINE__);
      result = 1;
    }

  /* Now with fseeko.  */
  if (fseeko (fp, 0, SEEK_SET) != 0)
    {
      printf ("%d: fseeko(fp, 0, SEEK_SET) failed\n", __LINE__);
      result = 1;
    }
  else if (fflush (fp) != 0)
    {
      printf ("%d: fflush() failed\n", __LINE__);
      result = 1;
    }
  else if (lseek (fd, 0, SEEK_CUR) != 0)
    {
      printf ("%d: lseek() returned different position\n", __LINE__);
      result = 1;
    }
  else if (fread (buf, sizeof (outstr) - 1, 1, fp) != 1)
    {
      printf ("%d: fread() failed\n", __LINE__);
      result = 1;
    }
  else if (memcmp (buf, outstr, sizeof (outstr) - 1) != 0)
    {
      printf ("%d: content after fseeko(,,SEEK_SET) wrong\n", __LINE__);
      result = 1;
    }

  /* Go back to the beginning of the file: relative.  */
  if (fseek (fp, -((int) sizeof (outstr) - 1), SEEK_CUR) != 0)
    {
      printf ("%d: fseek(fp, 0, SEEK_SET) failed\n", __LINE__);
      result = 1;
    }
  else if (fflush (fp) != 0)
    {
      printf ("%d: fflush() failed\n", __LINE__);
      result = 1;
    }
  else if (lseek (fd, 0, SEEK_CUR) != 0)
    {
      printf ("%d: lseek() returned different position\n", __LINE__);
      result = 1;
    }
  else if (fread (buf, sizeof (outstr) - 1, 1, fp) != 1)
    {
      printf ("%d: fread() failed\n", __LINE__);
      result = 1;
    }
  else if (memcmp (buf, outstr, sizeof (outstr) - 1) != 0)
    {
      printf ("%d: content after fseek(,,SEEK_SET) wrong\n", __LINE__);
      result = 1;
    }

  /* Now with fseeko.  */
  if (fseeko (fp, -((int) sizeof (outstr) - 1), SEEK_CUR) != 0)
    {
      printf ("%d: fseeko(fp, 0, SEEK_SET) failed\n", __LINE__);
      result = 1;
    }
  else if (fflush (fp) != 0)
    {
      printf ("%d: fflush() failed\n", __LINE__);
      result = 1;
    }
  else if (lseek (fd, 0, SEEK_CUR) != 0)
    {
      printf ("%d: lseek() returned different position\n", __LINE__);
      result = 1;
    }
  else if (fread (buf, sizeof (outstr) - 1, 1, fp) != 1)
    {
      printf ("%d: fread() failed\n", __LINE__);
      result = 1;
    }
  else if (memcmp (buf, outstr, sizeof (outstr) - 1) != 0)
    {
      printf ("%d: content after fseeko(,,SEEK_SET) wrong\n", __LINE__);
      result = 1;
    }

  /* Go back to the beginning of the file: from the end.  */
  if (fseek (fp, -((int) sizeof (outstr) - 1), SEEK_END) != 0)
    {
      printf ("%d: fseek(fp, 0, SEEK_SET) failed\n", __LINE__);
      result = 1;
    }
  else if (fflush (fp) != 0)
    {
      printf ("%d: fflush() failed\n", __LINE__);
      result = 1;
    }
  else if (lseek (fd, 0, SEEK_CUR) != 0)
    {
      printf ("%d: lseek() returned different position\n", __LINE__);
      result = 1;
    }
  else if (fread (buf, sizeof (outstr) - 1, 1, fp) != 1)
    {
      printf ("%d: fread() failed\n", __LINE__);
      result = 1;
    }
  else if (memcmp (buf, outstr, sizeof (outstr) - 1) != 0)
    {
      printf ("%d: content after fseek(,,SEEK_SET) wrong\n", __LINE__);
      result = 1;
    }

  /* Now with fseeko.  */
  if (fseeko (fp, -((int) sizeof (outstr) - 1), SEEK_END) != 0)
    {
      printf ("%d: fseeko(fp, 0, SEEK_SET) failed\n", __LINE__);
      result = 1;
    }
  else if (fflush (fp) != 0)
    {
      printf ("%d: fflush() failed\n", __LINE__);
      result = 1;
    }
  else if (lseek (fd, 0, SEEK_CUR) != 0)
    {
      printf ("%d: lseek() returned different position\n", __LINE__);
      result = 1;
    }
  else if (fread (buf, sizeof (outstr) - 1, 1, fp) != 1)
    {
      printf ("%d: fread() failed\n", __LINE__);
      result = 1;
    }
  else if (memcmp (buf, outstr, sizeof (outstr) - 1) != 0)
    {
      printf ("%d: content after fseeko(,,SEEK_SET) wrong\n", __LINE__);
      result = 1;
    }

  if (fwrite (outstr, sizeof (outstr) - 1, 1, fp) != 1)
    {
      printf ("%d: write error 2\n", __LINE__);
      result = 1;
      goto out;
    }

  if (fwrite (outstr, sizeof (outstr) - 1, 1, fp) != 1)
    {
      printf ("%d: write error 3\n", __LINE__);
      result = 1;
      goto out;
    }

  if (fwrite (outstr, sizeof (outstr) - 1, 1, fp) != 1)
    {
      printf ("%d: write error 4\n", __LINE__);
      result = 1;
      goto out;
    }

  if (fwrite (outstr, sizeof (outstr) - 1, 1, fp) != 1)
    {
      printf ("%d: write error 5\n", __LINE__);
      result = 1;
      goto out;
    }

  if (fputc ('1', fp) == EOF || fputc ('2', fp) == EOF)
    {
      printf ("%d: cannot add characters at the end\n", __LINE__);
      result = 1;
      goto out;
    }

  /* Check the access time.  */
  if (fstat64 (fd, &st1) < 0)
    {
      printf ("%d: fstat64() before fseeko() failed\n\n", __LINE__);
      result = 1;
    }
  else
    {
      sleep (1);

      if (fseek (fp, -(2 + 2 * (sizeof (outstr) - 1)), SEEK_CUR) != 0)
	{
	  printf ("%d: fseek() after write characters failed\n", __LINE__);
	  result = 1;
	  goto out;
	}
      else
	{

	  time_t t;
	  /* Make sure the timestamp actually can be different.  */
	  sleep (1);
	  t = time (NULL);

	  if (fstat64 (fd, &st2) < 0)
	    {
	      printf ("%d: fstat64() after fseeko() failed\n\n", __LINE__);
	      result = 1;
	    }
	  if (st1.st_ctime >= t)
	    {
	      printf ("%d: st_ctime not updated\n", __LINE__);
	      result = 1;
	    }
	  if (st1.st_mtime >= t)
	    {
	      printf ("%d: st_mtime not updated\n", __LINE__);
	      result = 1;
	    }
	  if (st1.st_ctime >= st2.st_ctime)
	    {
	      printf ("%d: st_ctime not changed\n", __LINE__);
	      result = 1;
	    }
	  if (st1.st_mtime >= st2.st_mtime)
	    {
	      printf ("%d: st_mtime not changed\n", __LINE__);
	      result = 1;
	    }
	}
    }

  if (fread (buf, 1, 2 + 2 * (sizeof (outstr) - 1), fp)
      != 2 + 2 * (sizeof (outstr) - 1))
    {
      printf ("%d: reading 2 records plus bits failed\n", __LINE__);
      result = 1;
    }
  else if (memcmp (buf, outstr, sizeof (outstr) - 1) != 0
	   || memcmp (&buf[sizeof (outstr) - 1], outstr,
		      sizeof (outstr) - 1) != 0
	   || buf[2 * (sizeof (outstr) - 1)] != '1'
	   || buf[2 * (sizeof (outstr) - 1) + 1] != '2')
    {
      printf ("%d: reading records failed\n", __LINE__);
      result = 1;
    }
  else if (ungetc ('9', fp) == EOF)
    {
      printf ("%d: ungetc() failed\n", __LINE__);
      result = 1;
    }
  else if (fseek (fp, -(2 + 2 * (sizeof (outstr) - 1)), SEEK_END) != 0)
    {
      printf ("%d: fseek after ungetc failed\n", __LINE__);
      result = 1;
    }
  else if (fread (buf, 1, 2 + 2 * (sizeof (outstr) - 1), fp)
      != 2 + 2 * (sizeof (outstr) - 1))
    {
      printf ("%d: reading 2 records plus bits failed\n", __LINE__);
      result = 1;
    }
  else if (memcmp (buf, outstr, sizeof (outstr) - 1) != 0
	   || memcmp (&buf[sizeof (outstr) - 1], outstr,
		      sizeof (outstr) - 1) != 0
	   || buf[2 * (sizeof (outstr) - 1)] != '1')
    {
      printf ("%d: reading records for the second time failed\n", __LINE__);
      result = 1;
    }
  else if (buf[2 * (sizeof (outstr) - 1) + 1] == '9')
    {
      printf ("%d: unget character not ignored\n", __LINE__);
      result = 1;
    }
  else if (buf[2 * (sizeof (outstr) - 1) + 1] != '2')
    {
      printf ("%d: unget somehow changed character\n", __LINE__);
      result = 1;
    }

  fclose (fp);

  fp = fopen (fname, "r");
  if (fp == NULL)
    {
      printf ("%d: fopen() failed\n\n", __LINE__);
      result = 1;
    }
  else if (fstat64 (fileno (fp), &st1) < 0)
    {
      printf ("%d: fstat64() before fseeko() failed\n\n", __LINE__);
      result = 1;
    }
  else if (fseeko (fp, 0, SEEK_END) != 0)
    {
      printf ("%d: fseeko(fp, 0, SEEK_END) failed\n", __LINE__);
      result = 1;
    }
  else if (ftello (fp) != st1.st_size)
    {
      printf ("%d: fstat64 st_size %zd ftello %zd\n", __LINE__,
	      (size_t) st1.st_size, (size_t) ftello (fp));
      result = 1;
    }
  else
    printf ("%d: SEEK_END works\n", __LINE__);
  if (fp != NULL)
    fclose (fp);

  fp = fopen (fname, "r");
  if (fp == NULL)
    {
      printf ("%d: fopen() failed\n\n", __LINE__);
      result = 1;
    }
  else if (fstat64 (fileno (fp), &st1) < 0)
    {
      printf ("%d: fstat64() before fgetc() failed\n\n", __LINE__);
      result = 1;
    }
  else if (fgetc (fp) == EOF)
    {
      printf ("%d: fgetc() before fseeko() failed\n\n", __LINE__);
      result = 1;
    }
  else if (fseeko (fp, 0, SEEK_END) != 0)
    {
      printf ("%d: fseeko(fp, 0, SEEK_END) failed\n", __LINE__);
      result = 1;
    }
  else if (ftello (fp) != st1.st_size)
    {
      printf ("%d: fstat64 st_size %zd ftello %zd\n", __LINE__,
	      (size_t) st1.st_size, (size_t) ftello (fp));
      result = 1;
    }
  else
    printf ("%d: SEEK_END works\n", __LINE__);
  if (fp != NULL)
    fclose (fp);

 out:
  unlink (fname);

  return result;
}

#define TEST_FUNCTION do_test ()
#include "../test-skeleton.c"
