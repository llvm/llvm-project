/* Stack executability handling for GNU dynamic linker.  Hurd version.
   Copyright (C) 2004-2021 Free Software Foundation, Inc.
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

#include <ldsodefs.h>
#include <hurdstartup.h>
#include <errno.h>

extern struct hurd_startup_data *_dl_hurd_data attribute_hidden;

/* There is no portable way to know the bounds of the initial thread's stack
   so as to mprotect it.  */

int
_dl_make_stack_executable (void **stack_endp)
{
  /* Challenge the caller.  */
  if (__builtin_expect (*stack_endp != __libc_stack_end, 0))
    return EPERM;
  *stack_endp = NULL;

#if IS_IN (rtld)
  if (__mprotect ((void *)_dl_hurd_data->stack_base, _dl_hurd_data->stack_size,
		  PROT_READ|PROT_WRITE|PROT_EXEC) != 0)
    return errno;

  /* Remember that we changed the permission.  */
  GL(dl_stack_flags) |= PF_X;

  return 0;
#else
  /* We don't bother to implement this for static linking.  */
  return ENOSYS;
#endif
}
rtld_hidden_def (_dl_make_stack_executable)
