/* Test re_search with dotless i.
   Copyright (C) 2006-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Jakub Jelinek <jakub@redhat.com>, 2006.

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
  struct re_pattern_buffer r;
  struct re_registers s;
  setlocale (LC_ALL, "en_US.UTF-8");
  memset (&r, 0, sizeof (r));
  memset (&s, 0, sizeof (s));
  re_set_syntax (RE_SYNTAX_GREP | RE_HAT_LISTS_NOT_NEWLINE | RE_ICASE);
  re_compile_pattern ("insert into", 11, &r);
  re_search (&r, "\xFF\0\x12\xA2\xAA\xC4\xB1,K\x12\xC4\xB1*\xACK",
	     15, 0, 15, &s);
  return 0;
}
