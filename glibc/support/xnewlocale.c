/* newlocale with error checking.
   Copyright (C) 2020-2021 Free Software Foundation, Inc.
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

#include <locale.h>

locale_t
xnewlocale (int category_mask, const char *locale, locale_t base)
{
  locale_t r = newlocale (category_mask, locale, base);
  if (r == (locale_t) 0)
    FAIL_EXIT1 ("error: newlocale (%d, \"%s\", %p)\n", category_mask,
		locale, base);
  return r;
}
