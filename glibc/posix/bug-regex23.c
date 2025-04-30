/* Test we don't segfault on invalid UTF-8 sequence.
   Copyright (C) 2004-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Jakub Jelinek <jakub@redhat.com>, 2004.

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

#include <locale.h>
#include <regex.h>
#include <string.h>

int
main (void)
{
  regex_t r;

  memset (&r, 0, sizeof (r));
  setlocale (LC_ALL, "de_DE.UTF-8");
  regcomp (&r, "[-a-z_0-9.]+@[-a-z_0-9.]+", REG_EXTENDED | REG_ICASE);
  regexec (&r, "\xe7\xb7\x95\xe7\x97", 0, NULL, 0);
  return 0;
}
