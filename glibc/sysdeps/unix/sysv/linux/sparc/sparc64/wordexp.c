/* Copyright (C) 2001-2021 Free Software Foundation, Inc.
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

#include <shlib-compat.h>

/* For Linux/Alpha we have to make the wordexp symbols versioned.  */
#define wordexp(words, pwordexp, flags) \
  __new_wordexp (words, pwordexp, flags)

#include <posix/wordexp.c>

versioned_symbol (libc, __new_wordexp, wordexp, GLIBC_2_2_2);


#if SHLIB_COMPAT (libc, GLIBC_2_0, GLIBC_2_2_2)
/* The old, incorrect wordexp_t definition.  */
typedef struct
  {
    int we_wordc;              /* Count of words matched.  */
    char **we_wordv;           /* List of expanded words.  */
    int we_offs;               /* Slots to reserve in `we_wordv'.  */
  } old_wordexp_t;


int
attribute_compat_text_section
__old_wordexp (const char *words, old_wordexp_t *pwordexp, int flags)
{
  wordexp_t we;
  int result;

  we.we_wordc = pwordexp->we_wordc;
  we.we_wordv = pwordexp->we_wordv;
  we.we_offs = pwordexp->we_offs;

  result = __new_wordexp (words, &we, flags);

  pwordexp->we_wordc = we.we_wordc;
  pwordexp->we_wordv = we.we_wordv;
  pwordexp->we_offs = we.we_offs;

  return result;
}
compat_symbol (libc, __old_wordexp, wordexp, GLIBC_2_1);
#endif
