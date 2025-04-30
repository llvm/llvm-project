/* Check cookie initialization for many auditors.  Auditor template.
   Copyright (C) 2019-2021 Free Software Foundation, Inc.
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

/* The macro MOD must be defined to the number of this auditor (an
   integer) before including this file.  */

#include <link.h>
#include <stdio.h>
#include <unistd.h>

/* Error counter for delayed error reporting.  */
static int errors;

unsigned int
la_version (unsigned int version)
{
  return version;
}

unsigned int
la_objopen (struct link_map *map, Lmid_t lmid,
            uintptr_t *cookie)
{
  struct link_map *cookie_map = (struct link_map *) *cookie;
  printf ("info: %d, la_objopen: map=%p name=%s cookie=%p:%p diff=%td\n",
          MOD, map, map->l_name, cookie, cookie_map,
          (char *) cookie - (char *) map);
  fflush (stdout);
  if (map != cookie_map)
    {
      printf ("error: %d, la_objopen:"
              " map address does not match cookie value\n",
              MOD);
      fflush (stdout);
      ++errors;
    }
  return 0;
}

extern unsigned int
la_objclose (uintptr_t *__cookie)
{
  if (errors != 0)
    {
      printf ("error: exiting due to previous errors");
      _exit (1);
    }
  return 0;
}
