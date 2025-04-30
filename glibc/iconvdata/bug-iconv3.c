/* Copyright (C) 2003-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Ulrich Drepper <drepper@redhat.com>, 2003.

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

#include <dlfcn.h>
#include <iconv.h>
#include <stdio.h>
#include <stdlib.h>
#include <gnu/lib-names.h>


static int
do_test (void)
{
  /* Get the iconv machinery initialized.  */
  (void) iconv_open ("ISO-8859-1", "ISO-8859-2");

  /* Dynamically load libpthread.  */
  if (dlopen (LIBPTHREAD_SO, RTLD_NOW) == NULL)
    {
      printf ("cannot load %s: %s\n", LIBPTHREAD_SO, dlerror ());
      exit (1);
    }

  /* And load some more.  This call hang for some configuration since
     the internal locking necessary wasn't adequately written to
     handle a dynamically loaded libpthread after the first call to
     iconv_open.  */
  (void) iconv_open ("ISO-8859-2", "ISO-8859-3");

  return 0;
}

#define TEST_FUNCTION do_test ()
#include "../test-skeleton.c"
