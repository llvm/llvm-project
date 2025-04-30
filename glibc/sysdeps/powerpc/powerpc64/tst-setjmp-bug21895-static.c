/* Test setjmp interoperability with static dlopen BZ #21895.
   Copyright (C) 2017-2021 Free Software Foundation, Inc.
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

#include <setjmp.h>
#include <stdio.h>
#include <stdlib.h>
#include <dlfcn.h>

/* Set TOC area pointed by sp to zero.  */
#define SET_TOC_TO_ZERO(sp)					 \
  ({								 \
    unsigned int zero = 0;					 \
    asm volatile ("std %0, 24(%1)\n\t" :: "r" (zero), "r" (sp)); \
  })

static void
bar (jmp_buf jb, unsigned long sp)
{
  static int i;
  if (i++==1)
    exit(0);	/* Success.  */

  /* This will set TOC are on caller frame (foo) to zero. __longjmp
     must restore r2 otherwise a segmentation fault will happens after
     it jumps back to foo.  */
  SET_TOC_TO_ZERO(sp);
  longjmp(jb, i);
}

static int
do_test (void)
{
  void *h = dlopen("setjmp-bug21895.so", RTLD_NOW);
  if (!h)
    {
      puts(dlerror());
      return 1;
    }

  void (*pfoo)(void) = dlsym(h, "foo");
  if (!pfoo)
    {
      puts(dlerror());
      return 1;
    }

  void (**ppbar)(jmp_buf, unsigned long) = dlsym(h, "bar");
  if (!ppbar)
    {
      puts(dlerror());
      return 1;
    }

  *ppbar = bar;
  pfoo();

  for(;;);
}

#include <support/test-driver.c>
