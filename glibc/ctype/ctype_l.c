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
  int __##name (int c, locale_t l) { return __isctype_l (c, type, l); } \
  weak_alias (__##name, name)

func (isalnum_l, _ISalnum)
func (isalpha_l, _ISalpha)
func (iscntrl_l, _IScntrl)
func (isdigit_l, _ISdigit)
func (islower_l, _ISlower)
func (isgraph_l, _ISgraph)
func (isprint_l, _ISprint)
func (ispunct_l, _ISpunct)
func (isspace_l, _ISspace)
func (isupper_l, _ISupper)
func (isxdigit_l, _ISxdigit)

int
(__tolower_l) (int c, locale_t l)
{
  return l->__ctype_tolower[c];
}
weak_alias (__tolower_l, tolower_l)

int
(__toupper_l) (int c, locale_t l)
{
  return l->__ctype_toupper[c];
}
weak_alias (__toupper_l, toupper_l)
