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

#include <nptl-stack.h>
#include <ldsodefs.h>
#include <pthreadP.h>

size_t __nptl_stack_cache_maxsize = 40 * 1024 * 1024;

void
__nptl_stack_list_del (list_t *elem)
{
  GL (dl_in_flight_stack) = (uintptr_t) elem;

  atomic_write_barrier ();

  list_del (elem);

  atomic_write_barrier ();

  GL (dl_in_flight_stack) = 0;
}
libc_hidden_def (__nptl_stack_list_del)

void
__nptl_stack_list_add (list_t *elem, list_t *list)
{
  GL (dl_in_flight_stack) = (uintptr_t) elem | 1;

  atomic_write_barrier ();

  list_add (elem, list);

  atomic_write_barrier ();

  GL (dl_in_flight_stack) = 0;
}
libc_hidden_def (__nptl_stack_list_add)

void
__nptl_free_stacks (size_t limit)
{
  /* We reduce the size of the cache.  Remove the last entries until
     the size is below the limit.  */
  list_t *entry;
  list_t *prev;

  /* Search from the end of the list.  */
  list_for_each_prev_safe (entry, prev, &GL (dl_stack_cache))
    {
      struct pthread *curr;

      curr = list_entry (entry, struct pthread, list);
      if (__nptl_stack_in_use (curr))
	{
	  /* Unlink the block.  */
	  __nptl_stack_list_del (entry);

	  /* Account for the freed memory.  */
	  GL (dl_stack_cache_actsize) -= curr->stackblock_size;

	  /* Free the memory associated with the ELF TLS.  */
	  _dl_deallocate_tls (TLS_TPADJ (curr), false);

	  /* Remove this block.  This should never fail.  If it does
	     something is really wrong.  */
	  if (__munmap (curr->stackblock, curr->stackblock_size) != 0)
	    abort ();

	  /* Maybe we have freed enough.  */
	  if (GL (dl_stack_cache_actsize) <= limit)
	    break;
	}
    }
}

/* Add a stack frame which is not used anymore to the stack.  Must be
   called with the cache lock held.  */
static inline void
__attribute ((always_inline))
queue_stack (struct pthread *stack)
{
  /* We unconditionally add the stack to the list.  The memory may
     still be in use but it will not be reused until the kernel marks
     the stack as not used anymore.  */
  __nptl_stack_list_add (&stack->list, &GL (dl_stack_cache));

  GL (dl_stack_cache_actsize) += stack->stackblock_size;
  if (__glibc_unlikely (GL (dl_stack_cache_actsize)
			> __nptl_stack_cache_maxsize))
    __nptl_free_stacks (__nptl_stack_cache_maxsize);
}

void
__nptl_deallocate_stack (struct pthread *pd)
{
  lll_lock (GL (dl_stack_cache_lock), LLL_PRIVATE);

  /* Remove the thread from the list of threads with user defined
     stacks.  */
  __nptl_stack_list_del (&pd->list);

  /* Not much to do.  Just free the mmap()ed memory.  Note that we do
     not reset the 'used' flag in the 'tid' field.  This is done by
     the kernel.  If no thread has been created yet this field is
     still zero.  */
  if (__glibc_likely (! pd->user_stack))
    (void) queue_stack (pd);
  else
    /* Free the memory associated with the ELF TLS.  */
    _dl_deallocate_tls (TLS_TPADJ (pd), false);

  lll_unlock (GL (dl_stack_cache_lock), LLL_PRIVATE);
}
libc_hidden_def (__nptl_deallocate_stack)

/* This function is internal (it has a GLIBC_PRIVATE) version, but it
   is widely used (either via weak symbol, or dlsym) to obtain the
   __static_tls_size value.  This value is then used to adjust the
   value of the stack size attribute, so that applications receive the
   full requested stack size, not diminished by the TCB and static TLS
   allocation on the stack.  Once the TCB is separately allocated,
   this function should be removed or renamed (if it is still
   necessary at that point).  */
size_t
__pthread_get_minstack (const pthread_attr_t *attr)
{
  return (GLRO(dl_pagesize) + __nptl_tls_static_size_for_stack ()
	  + PTHREAD_STACK_MIN);
}
libc_hidden_def (__pthread_get_minstack)
