/* sigaltstack wrappers.
   Copyright (C) 2019-2021 Free Software Foundation, Inc.
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

#include <support/xsignal.h>
#include <support/support.h>
#include <support/xunistd.h>
#include <support/check.h>

#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/param.h> /* roundup, MAX */

#ifndef MAP_NORESERVE
# define MAP_NORESERVE 0
#endif
#ifndef MAP_STACK
# define MAP_STACK 0
#endif

/* The "cookie" returned by xalloc_sigstack points to one of these
   structures.  */
struct sigstack_desc
{
  struct support_stack stack;
  stack_t alt_stack; /* The address and size of the stack itself.  */
  stack_t old_stack; /* The previous signal stack.  */
};

void *
xalloc_sigstack (size_t size)
{
  struct sigstack_desc *desc = xmalloc (sizeof (struct sigstack_desc));
  desc->stack = support_stack_alloc (size);
  desc->alt_stack.ss_sp    = desc->stack.stack;
  desc->alt_stack.ss_flags = 0;
  desc->alt_stack.ss_size  = desc->stack.size;

  if (sigaltstack (&desc->alt_stack, &desc->old_stack))
    FAIL_EXIT1 ("sigaltstack (new stack: sp=%p, size=%zu, flags=%u): %m\n",
                desc->alt_stack.ss_sp, desc->alt_stack.ss_size,
                desc->alt_stack.ss_flags);

  return desc;
}

void
xfree_sigstack (void *stack)
{
  struct sigstack_desc *desc = stack;

  if (sigaltstack (&desc->old_stack, 0))
    FAIL_EXIT1 ("sigaltstack (restore old stack: sp=%p, size=%zu, flags=%u): "
                "%m\n", desc->old_stack.ss_sp, desc->old_stack.ss_size,
                desc->old_stack.ss_flags);
  support_stack_free (&desc->stack);
  free (desc);
}

void
xget_sigstack_location (const void *stack, unsigned char **addrp, size_t *sizep)
{
  const struct sigstack_desc *desc = stack;
  *addrp = desc->alt_stack.ss_sp;
  *sizep = desc->alt_stack.ss_size;
}
