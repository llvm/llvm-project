/* Copyright (C) 2009-2021 Free Software Foundation, Inc.
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
#include <printf.h>
#include <stdlib.h>
#include <libc-lock.h>


/* Array of functions indexed by format character.  */
libc_freeres_ptr (printf_va_arg_function **__printf_va_arg_table)
  attribute_hidden;

__libc_lock_define_initialized (static, lock);

/* Last type allocated.  */
static int pa_next_type = PA_LAST;


int
__register_printf_type (printf_va_arg_function fct)
{
  int result = -1;
  __libc_lock_lock (lock);

  if (__printf_va_arg_table == NULL)
    {
      __printf_va_arg_table = (printf_va_arg_function **)
	calloc (0x100 - PA_LAST, sizeof (void *));
      if (__printf_va_arg_table == NULL)
	goto out;
    }

  if (pa_next_type == 0x100)
    __set_errno (ENOSPC);
  else
    {
      result = pa_next_type++;
      __printf_va_arg_table[result - PA_LAST] = fct;
    }

 out:
  __libc_lock_unlock (lock);

  return result;
}
weak_alias (__register_printf_type, register_printf_type)
