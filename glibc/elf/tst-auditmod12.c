/* Audit module for tst-audit12.
   Copyright (C) 2015-2021 Free Software Foundation, Inc.
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

#include <link.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

unsigned int
la_version (unsigned int version)
{
  return version;
}

char *
la_objsearch (const char *name, uintptr_t *cookie, unsigned int flag)
{
  const char target[] = "tst-audit12mod2.so";

  size_t namelen = strlen (name);
  if (namelen >= sizeof (target) - 1
      && strcmp (name + namelen - (sizeof (target) - 1), target) == 0)
    {
      return (char *) "$ORIGIN/tst-audit12mod3.so";
    }
  return (char *) name;
}
