/* Heuristic for recognizing DSO file names.
   Copyright (C) 2021 Free Software Foundation, Inc.
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

#include <stdbool.h>
#include <string.h>

/* Returns true if the file name looks like a DSO name.  */
static bool
_dl_is_dso (const char *name)
{
  /* Recognize lib*.so*, ld-*.so*, ld.so.*, ld64.so.*.  ld-*.so*
     matches both platform dynamic linker names like ld-linux.so.2,
     and versioned dynamic loader names like ld-2.12.so.  */
  return (((strncmp (name, "lib", 3) == 0 || strncmp (name, "ld-", 3) == 0)
           && strstr (name, ".so") != NULL)
          || strncmp (name, "ld.so.", 6) == 0
          || strncmp (name, "ld64.so.", 8) == 0);
}
