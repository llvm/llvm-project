/* Temporary file handling for tests.
   Copyright (C) 1998-2021 Free Software Foundation, Inc.
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

/* This is required to get an mkstemp which can create large files on
   some 32-bit platforms. */
#define _FILE_OFFSET_BITS 64

#include <support/temp_file.h>
#include <support/temp_file-internal.h>
#include <support/support.h>

#include <paths.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

/* List of temporary files.  */
static struct temp_name_list
{
  struct temp_name_list *next;
  char *name;
  pid_t owner;
} *temp_name_list;

/* Location of the temporary files.  Set by the test skeleton via
   support_set_test_dir.  The string is not be freed.  */
static const char *test_dir = _PATH_TMP;

void
add_temp_file (const char *name)
{
  struct temp_name_list *newp
    = (struct temp_name_list *) xcalloc (sizeof (*newp), 1);
  char *newname = strdup (name);
  if (newname != NULL)
    {
      newp->name = newname;
      newp->next = temp_name_list;
      newp->owner = getpid ();
      temp_name_list = newp;
    }
  else
    free (newp);
}

int
create_temp_file_in_dir (const char *base, const char *dir, char **filename)
{
  char *fname;
  int fd;

  fname = xasprintf ("%s/%sXXXXXX", dir, base);

  fd = mkstemp (fname);
  if (fd == -1)
    {
      printf ("cannot open temporary file '%s': %m\n", fname);
      free (fname);
      return -1;
    }

  add_temp_file (fname);
  if (filename != NULL)
    *filename = fname;
  else
    free (fname);

  return fd;
}

int
create_temp_file (const char *base, char **filename)
{
  return create_temp_file_in_dir (base, test_dir, filename);
}

char *
support_create_temp_directory (const char *base)
{
  char *path = xasprintf ("%s/%sXXXXXX", test_dir, base);
  if (mkdtemp (path) == NULL)
    {
      printf ("error: mkdtemp (\"%s\"): %m", path);
      exit (1);
    }
  add_temp_file (path);
  return path;
}

/* Helper functions called by the test skeleton follow.  */

void
support_set_test_dir (const char *path)
{
  test_dir = path;
}

void
support_delete_temp_files (void)
{
  pid_t pid = getpid ();
  while (temp_name_list != NULL)
    {
      /* Only perform the removal if the path was registed in the same
	 process, as identified by the PID.  (This assumes that the
	 parent process which registered the temporary file sticks
	 around, to prevent PID reuse.)  */
      if (temp_name_list->owner == pid)
	{
	  if (remove (temp_name_list->name) != 0)
	    printf ("warning: could not remove temporary file: %s: %m\n",
		    temp_name_list->name);
	}
      free (temp_name_list->name);

      struct temp_name_list *next = temp_name_list->next;
      free (temp_name_list);
      temp_name_list = next;
    }
}

void
support_print_temp_files (FILE *f)
{
  if (temp_name_list != NULL)
    {
      struct temp_name_list *n;
      fprintf (f, "temp_files=(\n");
      for (n = temp_name_list; n != NULL; n = n->next)
        fprintf (f, "  '%s'\n", n->name);
      fprintf (f, ")\n");
    }
}
