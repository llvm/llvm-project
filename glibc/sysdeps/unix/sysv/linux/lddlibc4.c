/* Stub for ldd script to print Linux libc4 dependencies.
   Copyright (C) 1998-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Ulrich Drepper <drepper@cygnus.com>, 1998.

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

/* This code is based on the `ldd' program code from the Linux ld.so
   package.  */

#include <a.out.h>
#include <errno.h>
#include <error.h>
#include <libintl.h>
#include <locale.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

/* Get libc version number.  */
#include "../version.h"

#define PACKAGE _libc_intl_domainname


int
main (int argc, char *argv[])
{
  const char *filename;
  size_t filename_len;
  struct exec exec;
  char *buf;
  FILE *fp;

  /* Set locale via LC_ALL.  */
  setlocale (LC_ALL, "");

  /* Set the text message domain.  */
  textdomain (PACKAGE);

  /* We expect exactly one argument.  */
  if (argc != 2)
    return 1;

  if (strcmp (argv[1], "--help") == 0)
    {
      printf (gettext ("Usage: lddlibc4 FILE\n\n"));
      printf (gettext ("For bug reporting instructions, please see:\n\
%s.\n"), REPORT_BUGS_TO);
      return 0;
    }
  else if (strcmp (argv[1], "--version") == 0)
    {
      printf ("lddlibc4 %s%s\n", PKGVERSION, VERSION);
      printf (gettext ("\
Copyright (C) %s Free Software Foundation, Inc.\n\
This is free software; see the source for copying conditions.  There is NO\n\
warranty; not even for MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.\n\
"), "2021");
      return 0;
    }

  filename = argv[1];

  /* First see whether this is really an a.out binary.  */
  fp = fopen (filename, "rb");
  if (fp == NULL)
    error (2, errno, gettext ("cannot open `%s'"), filename);

  /* Read the program header.  */
  if (fread (&exec, sizeof exec, 1, fp) < 1)
    error (2, errno, gettext ("cannot read header from `%s'"), filename);

  /* Test for the magic numbers.  */
  if (N_MAGIC (exec) != ZMAGIC && N_MAGIC (exec) != QMAGIC
      && N_MAGIC (exec) != OMAGIC)
    exit (3);

  /* We don't need the file open anymore.  */
  fclose (fp);

  /* We must put `__LDD_ARGV0=<program-name>' in the environment.  */
  filename_len = strlen (filename);
  buf = (char *) alloca (sizeof "__LDD_ARGV0=" + filename_len);
  mempcpy (mempcpy (buf, "__LDD_ARGV0=", sizeof "__LDD_ARGV0=" - 1),
	   filename, filename_len + 1);
  /* ...and put the value in the environment.  */
  putenv (buf);

  /* Now we can execute the binary.  */
  return execv (filename, &argv[argc]) ? 4 : 0;
}
