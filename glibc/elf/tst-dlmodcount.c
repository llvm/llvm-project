/* Copyright (C) 2004-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by David Mosberger <davidm@hpl.hp.com>, 2004.

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

#include <link.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>

#define SET	0
#define ADD	1
#define REMOVE	2

#define leq(l,r)	(((r) - (l)) <= ~0ULL / 2)

static int
callback (struct dl_phdr_info *info, size_t size, void *ptr)
{
  static int last_adds = 0, last_subs = 0;
  intptr_t cmd = (intptr_t) ptr;

  printf ("  size = %Zu\n", size);
  if (size < (offsetof (struct dl_phdr_info, dlpi_subs)
	      + sizeof (info->dlpi_subs)))
    {
      fprintf (stderr, "dl_iterate_phdr failed to pass dlpi_adds/dlpi_subs\n");
      exit (5);
    }

  printf ("  dlpi_adds = %Lu dlpi_subs = %Lu\n",
	  info->dlpi_adds, info->dlpi_subs);

  switch (cmd)
    {
    case SET:
      break;

    case ADD:
      if (leq (info->dlpi_adds, last_adds))
	{
	  fprintf (stderr, "dlpi_adds failed to get incremented!\n");
	  exit (3);
	}
      break;

    case REMOVE:
      if (leq (info->dlpi_subs, last_subs))
	{
	  fprintf (stderr, "dlpi_subs failed to get incremented!\n");
	  exit (4);
	}
      break;
    }
  last_adds = info->dlpi_adds;
  last_subs = info->dlpi_subs;
  return -1;
}

static void *
load (const char *path)
{
  void *handle;

  printf ("loading `%s'\n", path);
  handle = dlopen (path, RTLD_LAZY);
  if (!handle)
    exit (1);
  dl_iterate_phdr (callback, (void *)(intptr_t) ADD);
  return handle;
}

static void
unload (const char *path, void *handle)
{
  printf ("unloading `%s'\n", path);
  if (dlclose (handle) < 0)
    exit (2);
  dl_iterate_phdr (callback, (void *)(intptr_t) REMOVE);
}

static int
do_test (void)
{
  void *handle1, *handle2;

  dl_iterate_phdr (callback, (void *)(intptr_t) SET);
  handle1 = load ("firstobj.so");
  handle2 = load ("globalmod1.so");
  unload ("firstobj.so", handle1);
  unload ("globalmod1.so", handle2);
  return 0;
}

#include <support/test-driver.c>
