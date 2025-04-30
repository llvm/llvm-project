/* Basic test for scandir function.
   Copyright (C) 2015-2021 Free Software Foundation, Inc.
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

#include <stdbool.h>
#include <dirent.h>
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifndef D
# define D(x) x
#endif

static void prepare (void);
static int do_test (void);
#define PREPARE(argc, argv)     prepare ()
#define TEST_FUNCTION           do_test ()
#include "../test-skeleton.c"

static const char *scandir_test_dir;

static void
prepare (void)
{
  size_t test_dir_len = strlen (test_dir);
  static const char dir_name[] = "/tst-scandir.XXXXXX";

  size_t dirbuflen = test_dir_len + sizeof (dir_name);
  char *dirbuf = malloc (dirbuflen);
  if (dirbuf == NULL)
    {
      puts ("out of memory");
      exit (1);
    }

  snprintf (dirbuf, dirbuflen, "%s%s", test_dir, dir_name);
  if (mkdtemp (dirbuf) == NULL)
    {
      puts ("cannot create temporary directory");
      exit (1);
    }

  add_temp_file (dirbuf);
  scandir_test_dir = dirbuf;
}

/* The directory should be empty save the . and .. files.  */
static void
verify_empty (const char *dirname)
{
  DIR *dir = opendir (dirname);
  if (dir == NULL)
    {
      printf ("opendir (%s): %s\n", dirname, strerror (errno));
      exit (1);
    }

  struct dirent64 *d;
  while ((d = readdir64 (dir)) != NULL)
    if (strcmp (d->d_name, ".") != 0 && strcmp (d->d_name, "..") != 0)
      {
        printf ("temp directory contains file \"%s\"\n", d->d_name);
        exit (1);
      }

  closedir (dir);
}

static void
make_file (const char *dirname, const char *filename)
{
  char *name = NULL;
  if (asprintf (&name, "%s/%s", dirname, filename) < 0)
    {
      puts ("out of memory");
      exit (1);
    }

  int fd = open (name, O_WRONLY | O_CREAT | O_EXCL, 0600);
  if (fd < 0)
    {
      printf ("cannot create \"%s\": %s\n", name, strerror (errno));
      exit (1);
    }
  close (fd);

  free (name);
}

static void
remove_file (const char *dirname, const char *filename)
{
  char *name = NULL;
  if (asprintf (&name, "%s/%s", dirname, filename) < 0)
    {
      puts ("out of memory");
      exit (1);
    }

  remove (name);

  free (name);
}

static void
freelist (struct D(dirent) **list, size_t n)
{
  for (size_t i = 0; i < n; ++i)
    free (list[i]);
  free (list);
}

static int
select_a (const struct D(dirent) *d)
{
  return d->d_name[0] == 'a';
}

static int
do_test (void)
{
  verify_empty (scandir_test_dir);

  make_file (scandir_test_dir, "c");
  make_file (scandir_test_dir, "aa");
  make_file (scandir_test_dir, "b");
  make_file (scandir_test_dir, "a");


  /* First a basic test with no select or compare functions.  */

  struct D(dirent) **list;
  int n = D(scandir) (scandir_test_dir, &list, NULL, NULL);
  if (n < 0)
    {
      printf ("scandir failed on %s: %s\n",
              scandir_test_dir, strerror (errno));
      return 1;
    }
  if (n != 6)
    {
      printf ("scandir returned %d entries instead of 6\n", n);
      return 1;
    }

  struct
  {
    const char *name;
    bool found;
  } expected[] =
    {
      { ".", },
      { "..", },
      { "a", },
      { "aa", },
      { "b", },
      { "c", },
    };

  /* Verify the results, ignoring the order.  */
  for (int i = 0; i < n; ++i)
    {
      for (size_t j = 0; j < sizeof expected / sizeof expected[0]; ++j)
        if (!strcmp (list[i]->d_name, expected[j].name))
          {
            expected[j].found = true;
            goto found;
          }

      printf ("scandir yields unexpected entry [%d] \"%s\"\n",
              i, list[i]->d_name);
      return 1;

    found:;
    }

  for (size_t j = 0; j < sizeof expected / sizeof expected[0]; ++j)
    if (!expected[j].found)
      {
        printf ("failed to produce \"%s\"\n", expected[j].name);
        return 1;
      }

  freelist (list, n);


  /* Now a test with a comparison function.  */

  n = D(scandir) (scandir_test_dir, &list, NULL, &D(alphasort));
  if (n < 0)
    {
      printf ("scandir failed on %s: %s\n",
              scandir_test_dir, strerror (errno));
      return 1;
    }
  if (n != 6)
    {
      printf ("scandir returned %d entries instead of 6\n", n);
      return 1;
    }

  assert (sizeof expected / sizeof expected[0] == 6);
  for (int i = 0; i < n; ++i)
    if (strcmp (list[i]->d_name, expected[i].name))
      {
        printf ("scandir yields [%d] of \"%s\", expected \"%s\"\n",
                i, list[i]->d_name, expected[i].name);
        return 1;
      }

  freelist (list, n);


  /* Now a test with a select function but no comparison function.  */

  n = D(scandir) (scandir_test_dir, &list, &select_a, NULL);
  if (n < 0)
    {
      printf ("scandir failed on %s: %s\n",
              scandir_test_dir, strerror (errno));
      return 1;
    }
  if (n != 2)
    {
      printf ("scandir returned %d entries instead of 2\n", n);
      return 1;
    }

  if (strcmp (list[0]->d_name, "a") && strcmp (list[0]->d_name, "aa"))
    {
      printf ("scandir yields [0] \"%s\", expected \"a\" or \"aa\"\n",
              list[0]->d_name);
      return 1;
    }
  if (strcmp (list[1]->d_name, "a") && strcmp (list[1]->d_name, "aa"))
    {
      printf ("scandir yields [1] \"%s\", expected \"a\" or \"aa\"\n",
              list[1]->d_name);
      return 1;
    }
  if (!strcmp (list[0]->d_name, list[1]->d_name))
    {
      printf ("scandir yields \"%s\" twice!\n", list[0]->d_name);
      return 1;
    }

  freelist (list, n);


  /* Now a test with both functions.  */

  n = D(scandir) (scandir_test_dir, &list, &select_a, &D(alphasort));
  if (n < 0)
    {
      printf ("scandir failed on %s: %s\n",
              scandir_test_dir, strerror (errno));
      return 1;
    }
  if (n != 2)
    {
      printf ("scandir returned %d entries instead of 2\n", n);
      return 1;
    }

  if (strcmp (list[0]->d_name, "a") || strcmp (list[1]->d_name, "aa"))
    {
      printf ("scandir yields {\"%s\", \"%s\"}, expected {\"a\", \"aa\"}\n",
              list[0]->d_name, list[1]->d_name);
      return 1;
    }

  freelist (list, n);


  /* Clean up the test directory.  */
  remove_file (scandir_test_dir, "c");
  remove_file (scandir_test_dir, "aa");
  remove_file (scandir_test_dir, "b");
  remove_file (scandir_test_dir, "a");

  return 0;
}
