/* Copyright (C) 1996-2021 Free Software Foundation, Inc.
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

#include <ctype.h>
#include <wctype.h>
#include <stdint.h>
#include <locale.h>
#include <locale/localeinfo.h>

#define USE_IN_EXTENDED_LOCALE_MODEL
#include "wchar-lookup.h"

/* Provide real-function versions of all the wctype macros.  */

#define	func(name, type) \
  int __isw##name (wint_t wc, locale_t locale)				      \
  {									      \
    if (isascii (wc))							      \
      return is##name ((int) wc, locale);				      \
    size_t i = locale->__locales[LC_CTYPE]->values[_NL_ITEM_INDEX (_NL_CTYPE_CLASS_OFFSET)].word + type; \
    const char *desc = locale->__locales[LC_CTYPE]->values[i].string;	      \
    return wctype_table_lookup (desc, wc);				      \
  }									      \
  libc_hidden_def (__isw##name)						      \
  weak_alias (__isw##name, isw##name)

func (alnum_l, __ISwalnum)
func (alpha_l, __ISwalpha)
func (blank_l, __ISwblank)
func (cntrl_l, __ISwcntrl)
#undef iswdigit_l
#undef __iswdigit_l
func (digit_l, __ISwdigit)
func (lower_l, __ISwlower)
func (graph_l, __ISwgraph)
func (print_l, __ISwprint)
func (punct_l, __ISwpunct)
func (space_l, __ISwspace)
func (upper_l, __ISwupper)
func (xdigit_l, __ISwxdigit)

wint_t
(__towlower_l) (wint_t wc, locale_t locale)
{
  size_t i = locale->__locales[LC_CTYPE]->values[_NL_ITEM_INDEX (_NL_CTYPE_MAP_OFFSET)].word + __TOW_tolower;
  const char *desc = locale->__locales[LC_CTYPE]->values[i].string;
  return wctrans_table_lookup (desc, wc);
}
libc_hidden_def (__towlower_l)
weak_alias (__towlower_l, towlower_l)

wint_t
(__towupper_l) (wint_t wc, locale_t locale)
{
  size_t i = locale->__locales[LC_CTYPE]->values[_NL_ITEM_INDEX (_NL_CTYPE_MAP_OFFSET)].word + __TOW_toupper;
  const char *desc = locale->__locales[LC_CTYPE]->values[i].string;
  return wctrans_table_lookup (desc, wc);
}
libc_hidden_def (__towupper_l)
weak_alias (__towupper_l, towupper_l)
