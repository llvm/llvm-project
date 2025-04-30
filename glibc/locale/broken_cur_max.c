/* Return number of characters in multibyte representation for current
   character set.
   Copyright (C) 1996-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Ulrich Drepper <drepper@cygnus.com>, 1996.

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

#include <langinfo.h>
#include <locale.h>
#include <stdlib.h>
#include "localeinfo.h"


/* This is a gross hack to get broken programs running.

   ISO C provides no mean to find out how many bytes the wide
   character representation really uses.  But it defines MB_CUR_LEN to
   return the information for the multi-byte character representation.
   Many programmers don't know the difference between the two and
   think this means the same.  But assuming all characters have a size
   of MB_CUR_LEN after they have been processed by `mbrtowc' is wrong.
   Instead the maximum number of characters used for the conversion is
   MB_CUR_LEN.

   It is known that some Motif applications have this problem.  To
   cure this one has to make sure the glibc uses the function in this
   file instead of the one in locale/mb_cur_max.c.  This can either be
   done by linking with this file or by using the LD_PRELOAD feature
   of the dynamic linker.  */
size_t
__ctype_get_mb_cur_max (void)
{
  union locale_data_value u;

  u.string = nl_langinfo (_NL_CTYPE_MB_CUR_MAX);
  return ((size_t []) { 1, 1, 1, 2, 2, 3, 4 })[u.word];
}
