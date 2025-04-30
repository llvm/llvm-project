/* basic fmemopen interface testing.
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

#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>

static char *test_file;

static void
do_prepare (int argc, char *argv[])
{
  /* Construct the test file name based on ARGV[0], which will be
     an absolute file name in the build directory.  Don't touch the
     source directory, which might be read-only.  */
  if (asprintf (&test_file, "%s.test", argv[0]) < 0)
    {
      puts ("asprintf failed\n");
      exit (EXIT_FAILURE);
    }
}

static int
do_test (void)
{
  const char blah[] = "BLAH";
  FILE *fp;
  char *mmap_data;
  int ch, fd;
  struct stat fs;
  const char *cp;

  /* setup the physical file, and use it */
  if ((fp = fopen (test_file, "w+")) == NULL)
    return 1;
  if (fwrite (blah, 1, strlen (blah), fp) != strlen (blah))
    {
      fclose (fp);
      return 2;
    }

  rewind (fp);
  printf ("file: ");
  cp = blah;
  while ((ch = getc (fp)) != EOF)
    {
      fputc (ch, stdout);
      if (ch != *cp)
	{
	  printf ("\ncharacter %td: '%c' instead of '%c'\n",
		  cp - blah, ch, *cp);
	  fclose (fp);
	  return 1;
	}
      ++cp;
    }
  fputc ('\n', stdout);
  if (ferror (fp))
    {
      puts ("fp: error");
      fclose (fp);
      return 1;
    }
  if (feof (fp))
    printf ("fp: EOF\n");
  else
    {
      puts ("not EOF");
      fclose (fp);
      return 1;
    }
  fclose (fp);

  /* Now, mmap the file into a buffer, and do that too */
  if ((fd = open (test_file, O_RDONLY)) == -1)
    {
      printf ("open (%s, O_RDONLY) failed\n", test_file);
      return 3;
    }
  if (fstat (fd, &fs) == -1)
    {
      printf ("stat (%i)\n", fd);
      return 4;
    }

  if ((mmap_data = (char *) mmap (NULL, fs.st_size, PROT_READ,
				  MAP_SHARED, fd, 0)) == MAP_FAILED)
    {
      printf ("mmap (NULL, %zu, PROT_READ, MAP_SHARED, %i, 0) failed\n",
	      (size_t) fs.st_size, fd);
      return 5;
    }

  if ((fp = fmemopen (mmap_data, fs.st_size, "r")) == NULL)
    {
      printf ("fmemopen (%p, %zu) failed\n", mmap_data, (size_t) fs.st_size);
      return 1;
    }

  printf ("mem: ");
  cp = blah;
  while ((ch = getc (fp)) != EOF)
    {
      fputc (ch, stdout);
      if (ch != *cp)
	{
	  printf ("%td character: '%c' instead of '%c'\n",
		  cp - blah, ch, *cp);
	  fclose (fp);
	  return 1;
	}
      ++cp;
    }

  fputc ('\n', stdout);

  if (ferror (fp))
    {
      puts ("fp: error");
      fclose (fp);
      return 1;
    }
  if (feof (fp))
    printf ("fp: EOF\n");
  else
    {
      puts ("not EOF");
      fclose (fp);
      return 1;
    }

  fclose (fp);

  munmap (mmap_data, fs.st_size);

  unlink (test_file);
  free (test_file);

  return 0;
}

#define PREPARE(argc, argv) do_prepare (argc, argv)
#define TEST_FUNCTION       do_test ()
#include "../test-skeleton.c"
