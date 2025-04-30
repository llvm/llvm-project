/* Frees the dynamically allocated storage from an earlier call to glob.
   Copyright (C) 2017-2021 Free Software Foundation, Inc.
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

#ifndef _LIBC
# include <libc-config.h>
#endif

#include <glob.h>
#include <stdlib.h>

/* Free storage allocated in PGLOB by a previous `glob' call.  */
void
globfree (glob_t *pglob)
{
  if (pglob->gl_pathv != NULL)
    {
      size_t i;
      for (i = 0; i < pglob->gl_pathc; ++i)
        free (pglob->gl_pathv[pglob->gl_offs + i]);
      free (pglob->gl_pathv);
      pglob->gl_pathv = NULL;
    }
}
#ifndef globfree
libc_hidden_def (globfree)
#endif
