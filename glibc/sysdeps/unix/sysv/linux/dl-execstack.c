/* Stack executability handling for GNU dynamic linker.  Linux version.
   Copyright (C) 2003-2021 Free Software Foundation, Inc.
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

#include <errno.h>
#include <ldsodefs.h>
#include <libintl.h>
#include <list.h>
#include <pthreadP.h>
#include <stackinfo.h>
#include <stdbool.h>
#include <sys/mman.h>
#include <sysdep.h>
#include <unistd.h>

extern int __stack_prot attribute_relro attribute_hidden;

static int
make_main_stack_executable (void **stack_endp)
{
  /* This gives us the highest/lowest page that needs to be changed.  */
  uintptr_t page = ((uintptr_t) *stack_endp
		    & -(intptr_t) GLRO(dl_pagesize));
  int result = 0;

  if (__builtin_expect (__mprotect ((void *) page, GLRO(dl_pagesize),
				    __stack_prot) == 0, 1))
    goto return_success;
  result = errno;
  goto out;

 return_success:
  /* Clear the address.  */
  *stack_endp = NULL;

  /* Remember that we changed the permission.  */
  GL(dl_stack_flags) |= PF_X;

 out:
#ifdef check_consistency
  check_consistency ();
#endif

  return result;
}

int
_dl_make_stacks_executable (void **stack_endp)
{
  /* First the main thread's stack.  */
  int err = make_main_stack_executable (stack_endp);
  if (err != 0)
    return err;

  lll_lock (GL (dl_stack_cache_lock), LLL_PRIVATE);

  list_t *runp;
  list_for_each (runp, &GL (dl_stack_used))
    {
      err = __nptl_change_stack_perm (list_entry (runp, struct pthread, list));
      if (err != 0)
	break;
    }

  /* Also change the permission for the currently unused stacks.  This
     might be wasted time but better spend it here than adding a check
     in the fast path.  */
  if (err == 0)
    list_for_each (runp, &GL (dl_stack_cache))
      {
	err = __nptl_change_stack_perm (list_entry (runp, struct pthread,
						    list));
	if (err != 0)
	  break;
      }

  lll_unlock (GL (dl_stack_cache_lock), LLL_PRIVATE);

  return err;
}

int
__nptl_change_stack_perm (struct pthread *pd)
{
#ifdef NEED_SEPARATE_REGISTER_STACK
  size_t pagemask = __getpagesize () - 1;
  void *stack = (pd->stackblock
		 + (((((pd->stackblock_size - pd->guardsize) / 2)
		      & pagemask) + pd->guardsize) & pagemask));
  size_t len = pd->stackblock + pd->stackblock_size - stack;
#elif _STACK_GROWS_DOWN
  void *stack = pd->stackblock + pd->guardsize;
  size_t len = pd->stackblock_size - pd->guardsize;
#elif _STACK_GROWS_UP
  void *stack = pd->stackblock;
  size_t len = (uintptr_t) pd - pd->guardsize - (uintptr_t) pd->stackblock;
#else
# error "Define either _STACK_GROWS_DOWN or _STACK_GROWS_UP"
#endif
  if (__mprotect (stack, len, PROT_READ | PROT_WRITE | PROT_EXEC) != 0)
    return errno;

  return 0;
}
rtld_hidden_def (__nptl_change_stack_perm)
