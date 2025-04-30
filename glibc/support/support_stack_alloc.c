/* Allocate a stack suitable to be used with xclone or xsigaltstack.
   Copyright (C) 2021 Free Software Foundation, Inc.
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

#include <support/check.h>
#include <support/support.h>
#include <support/xunistd.h>
#include <stdint.h>
#include <string.h>
#include <stackinfo.h>
#include <sys/mman.h>
#include <sys/param.h> /* roundup, MAX  */

#ifndef MAP_NORESERVE
# define MAP_NORESERVE 0
#endif
#ifndef MAP_STACK
# define MAP_STACK 0
#endif

struct support_stack
support_stack_alloc (size_t size)
{
  size_t pagesize = sysconf (_SC_PAGESIZE);
  if (pagesize == -1)
    FAIL_EXIT1 ("sysconf (_SC_PAGESIZE): %m\n");

  /* Always supply at least sysconf (_SC_SIGSTKSZ) space; passing 0
     as size means only that much space.  No matter what the number is,
     round it up to a whole number of pages.  */
  size_t stacksize = roundup (size + sysconf (_SC_SIGSTKSZ),
			      pagesize);

  /* The guard bands need to be large enough to intercept offset
     accesses from a stack address that might otherwise hit another
     mapping.  Make them at least twice as big as the stack itself, to
     defend against an offset by the entire size of a large
     stack-allocated array.  The minimum is 1MiB, which is arbitrarily
     chosen to be larger than any "typical" wild pointer offset.
     Again, no matter what the number is, round it up to a whole
     number of pages.  */
  size_t guardsize = roundup (MAX (2 * stacksize, 1024 * 1024), pagesize);
  size_t alloc_size = guardsize + stacksize + guardsize;
  /* Use MAP_NORESERVE so that RAM will not be wasted on the guard
     bands; touch all the pages of the actual stack before returning,
     so we know they are allocated.  */
  void *alloc_base = xmmap (0,
                            alloc_size,
                            PROT_NONE,
                            MAP_PRIVATE|MAP_ANONYMOUS|MAP_NORESERVE|MAP_STACK,
                            -1);
  /* Some architecture still requires executable stack for the signal return
     trampoline, although PF_X could be overridden if PT_GNU_STACK is present.
     However since glibc does not export such information with a proper ABI,
     it uses the historical permissions.  */
  int prot = PROT_READ | PROT_WRITE
	     | (DEFAULT_STACK_PERMS & PF_X ? PROT_EXEC : 0);
  xmprotect (alloc_base + guardsize, stacksize, prot);
  memset (alloc_base + guardsize, 0xA5, stacksize);
  return (struct support_stack) { alloc_base + guardsize, stacksize, guardsize };
}

void
support_stack_free (struct support_stack *stack)
{
  void *alloc_base = (void *)((uintptr_t) stack->stack - stack->guardsize);
  size_t alloc_size = stack->size + 2 * stack->guardsize;
  xmunmap (alloc_base, alloc_size);
}
