/* Copyright (C) 1991-2021 Free Software Foundation, Inc.
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
#include <limits.h>
#include <printf.h>
#include <stddef.h>
#include <stdlib.h>
#include <libc-lock.h>


/* Array of functions indexed by format character.  */
libc_freeres_ptr (printf_arginfo_size_function **__printf_arginfo_table)
  attribute_hidden;
printf_function **__printf_function_table attribute_hidden;

__libc_lock_define_initialized (static, lock)

int __register_printf_specifier (int, printf_function,
				 printf_arginfo_size_function);
libc_hidden_proto (__register_printf_specifier)
int __register_printf_function (int, printf_function,
				printf_arginfo_function);


/* Register FUNC to be called to format SPEC specifiers.  */
int
__register_printf_specifier (int spec, printf_function converter,
			     printf_arginfo_size_function arginfo)
{
  if (spec < 0 || spec > (int) UCHAR_MAX)
    {
      __set_errno (EINVAL);
      return -1;
    }

  int result = 0;
  __libc_lock_lock (lock);

  if (__printf_function_table == NULL)
    {
      __printf_arginfo_table = (printf_arginfo_size_function **)
	calloc (UCHAR_MAX + 1, sizeof (void *) * 2);
      if (__printf_arginfo_table == NULL)
	{
	  result = -1;
	  goto out;
	}

      __printf_function_table = (printf_function **)
	(__printf_arginfo_table + UCHAR_MAX + 1);
    }

  __printf_function_table[spec] = converter;
  __printf_arginfo_table[spec] = arginfo;

 out:
  __libc_lock_unlock (lock);

  return result;
}
libc_hidden_def (__register_printf_specifier)
weak_alias (__register_printf_specifier, register_printf_specifier)


/* Register FUNC to be called to format SPEC specifiers.  */
int
__register_printf_function (int spec, printf_function converter,
			    printf_arginfo_function arginfo)
{
  return __register_printf_specifier (spec, converter,
				      (printf_arginfo_size_function*) arginfo);
}
weak_alias (__register_printf_function, register_printf_function)
