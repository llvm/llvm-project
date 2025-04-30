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

#define	__NO_CTYPE
#include <ctype.h>

/* Provide real-function versions of all the ctype macros.  */

#define	func(name, type) \
  int name (int c) { return __isctype (c, type); }

func (isalnum, _ISalnum)
func (isalpha, _ISalpha)
func (iscntrl, _IScntrl)
func (isdigit, _ISdigit)
func (islower, _ISlower)
func (isgraph, _ISgraph)
func (isprint, _ISprint)
func (ispunct, _ISpunct)
func (isspace, _ISspace)
func (isupper, _ISupper)
func (isxdigit, _ISxdigit)

#define __ctype_tolower \
  ((int32_t *) _NL_CURRENT (LC_CTYPE, _NL_CTYPE_TOLOWER) + 128)
#define __ctype_toupper \
  ((int32_t *) _NL_CURRENT (LC_CTYPE, _NL_CTYPE_TOUPPER) + 128)

int
tolower (int c)
{
  return c >= -128 && c < 256 ? __ctype_tolower[c] : c;
}
libc_hidden_def (tolower)

int
toupper (int c)
{
  return c >= -128 && c < 256 ? __ctype_toupper[c] : c;
}
libc_hidden_def (toupper)
