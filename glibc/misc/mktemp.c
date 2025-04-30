/* Copyright (C) 1998-2021 Free Software Foundation, Inc.
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

#include <stdio.h>
#include <stdlib.h>

/* Generate a unique temporary file name from TEMPLATE.
   The last six characters of TEMPLATE must be "XXXXXX";
   they are replaced with a string that makes the filename unique.  */
char *
__mktemp (char *template)
{
  if (__gen_tempname (template, 0, 0, __GT_NOCREATE) < 0)
    /* We return the null string if we can't find a unique file name.  */
    template[0] = '\0';

  return template;
}
libc_hidden_def (__mktemp)
weak_alias (__mktemp, mktemp)

link_warning (mktemp, "the use of `mktemp' is dangerous, "
		      "better use `mkstemp' or `mkdtemp'")
