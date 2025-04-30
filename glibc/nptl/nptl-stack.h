/* Stack cache management for NPTL.
   Copyright (C) 2002-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Ulrich Drepper <drepper@redhat.com>, 2002.

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

#ifndef _NPTL_STACK_H
#define _NPTL_STACK_H

#include <nptl/descr.h>
#include <ldsodefs.h>
#include <list.h>
#include <stdbool.h>

/* Maximum size of the cache, in bytes.  40 MiB by default.  */
extern size_t __nptl_stack_cache_maxsize attribute_hidden;

/* Check whether the stack is still used or not.  */
static inline bool
__nptl_stack_in_use (struct pthread *pd)
{
  return pd->tid <= 0;
}

/* Remove the stack ELEM from its list.  */
void __nptl_stack_list_del (list_t *elem);
libc_hidden_proto (__nptl_stack_list_del)

/* Add ELEM to a stack list.  LIST can be either &GL (dl_stack_used)
   or &GL (dl_stack_cache).  */
void __nptl_stack_list_add (list_t *elem, list_t *list);
libc_hidden_proto (__nptl_stack_list_add)

/* Free allocated stack.  */
extern void __nptl_deallocate_stack (struct pthread *pd);
libc_hidden_proto (__nptl_deallocate_stack)

/* Free stacks until cache size is lower than LIMIT.  */
void __nptl_free_stacks (size_t limit) attribute_hidden;

/* Compute the size of the static TLS area based on data from the
   dynamic loader.  */
static inline size_t
__nptl_tls_static_size_for_stack (void)
{
  return roundup (GLRO (dl_tls_static_size), GLRO (dl_tls_static_align));
}

#endif /* _NPTL_STACK_H */
