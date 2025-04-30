/* Internal header for proving correct grouping in strings of numbers.
   Copyright (C) 1995-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Ulrich Drepper <drepper@gnu.ai.mit.edu>, 1995.

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

/* Find the maximum prefix of the string between BEGIN and END which
   satisfies the grouping rules.  It is assumed that at least one digit
   follows BEGIN directly.  */
extern const wchar_t *__correctly_grouped_prefixwc (const wchar_t *begin,
						    const wchar_t *end,
						    wchar_t thousands,
						    const char *grouping)
     attribute_hidden;

extern const char *__correctly_grouped_prefixmb (const char *begin,
						 const char *end,
						 const char *thousands,
						 const char *grouping)
     attribute_hidden;
