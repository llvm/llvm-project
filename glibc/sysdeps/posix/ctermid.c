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

#include <stdio.h>
#include <string.h>


/* Return the name of the controlling terminal.  If S is not NULL, the
   name is copied into it (it should be at least L_ctermid bytes
   long), otherwise we return a pointer to a non-const but read-only
   string literal, that POSIX states the caller must not modify.  */
char *
ctermid (char *s)
{
  char *name = (char /*drop const*/ *) "/dev/tty";

  if (s == NULL)
    return name;

  return strcpy (s, name);
}
