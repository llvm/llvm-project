/* Copyright (C) 1996-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Ulrich Drepper <drepper@gnu.ai.mit.edu>, 1996.

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

#include <string.h>
#include <wchar.h>

#undef mbsinit
#undef __mbsinit

/* In GNU libc the conversion functions only can convert between the
   fixed wide character representation and the multibyte
   representation of the same character set.  Since we use ISO 10646
   in UCS4 encoding for wide characters the best solution for
   multibyte characters is the UTF8 encoding.  I.e., the only state
   information is a counter of the processed bytes so far and the
   value collected so far.  Especially, we don't have different shift
   states.  */
int
__mbsinit (const mbstate_t *ps)
{
  return ps == NULL || ps->__count == 0;
}
weak_alias (__mbsinit, mbsinit)
