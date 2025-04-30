/* Test that lazy binding does not clobber r0.
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

#include <assert.h>
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#if defined (__s390x__)
static const unsigned long magic_value = 0x0011223344556677UL;
#else
static const unsigned long magic_value = 0x00112233;
#endif

unsigned long r0x2_trampoline (unsigned long);

/* Invoke r0x2, which doubles the value in r0.  If we get
   value * 2 back, this means nothing clobbers r0, particularly,
   _dl_runtime_resolve and _dl_runtime_profile.  */
asm ("    .type r0x2_trampoline, @function\n"
     "r0x2_trampoline:\n"
#if defined (__s390x__)
     "    lgr %r0,%r2\n"
     "    stg %r14,112(%r15)\n"
     "    aghi %r15,-160\n"
     "    brasl %r14,r0x2@plt\n"
     "    aghi %r15,160\n"
     "    lg %r14,112(%r15)\n"
     "    lgr %r2,%r0\n"
     "    br %r14\n"
#elif defined (__zarch__)
     "    lr %r0,%r2\n"
     "    st %r14,56(%r15)\n"
     "    ahi %r15,-96\n"
     "    brasl %r14,r0x2@plt\n"
     "    ahi %r15,96\n"
     "    l %r14,56(%r15)\n"
     "    lr %r2,%r0\n"
     "    br %r14\n"
#else
     "    lr %r0,%r2\n"
     "    st %r14,56(%r15)\n"
     "    ahi %r15,-96\n"
     "    balr %r14,0\n"
     "    l %r14,1f-.(%r14)\n"
     "    basr %r14,%r14\n"
     "    ahi %r15,96\n"
     "    l %r14,56(%r15)\n"
     "    lr %r2,%r0\n"
     "    br %r14\n"
     "1: .long r0x2\n"
#endif
     );

static int
do_test (void)
{
  int i;
  unsigned long r0;
  const char *run;

  for (i = 0; i < 2; i++)
    {
      run = (i == 0) ? "lazy" : "non-lazy";
      r0 = magic_value;
      printf ("-> %s r0 = 0x%lx\n", run, r0);
      r0 = r0x2_trampoline (r0);
      printf ("<- %s r0 * 2 = 0x%lx\n", run, r0);
      if (r0 != magic_value * 2)
	return EXIT_FAILURE;
    }
  return EXIT_SUCCESS;
}

#include <support/test-driver.c>
