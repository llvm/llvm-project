/* Wrapper for printf_size.  IEEE128 version.
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

#include <printf.h>

extern __typeof (printf_size) __printf_size;

int
___ieee128_printf_size (FILE *fp, const struct printf_info *info,
			const void *const *args)
{
  struct printf_info info_ieee128 = *info;

  info_ieee128.is_binary128 = info->is_long_double;
  return __printf_size (fp, &info_ieee128, args);
}
strong_alias (___ieee128_printf_size, __printf_sizeieee128)
