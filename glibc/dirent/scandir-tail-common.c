/* Common implementation for scandir{at}.
   Copyright (C) 2018-2021 Free Software Foundation, Inc.
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

#include <string.h>
#include <errno.h>

int
SCANDIR_TAIL (DIR *dp,
              DIRENT_TYPE ***namelist,
              int (*select) (const DIRENT_TYPE *),
              int (*cmp) (const DIRENT_TYPE **, const DIRENT_TYPE **))
{
  if (dp == NULL)
    return -1;

  int save = errno;
  __set_errno (0);

  int result;
  struct scandir_cancel_struct c = { .dp = dp };
  __libc_cleanup_push (&__scandir_cancel_handler, &c);

  DIRENT_TYPE **v = NULL;
  size_t vsize = 0;
  DIRENT_TYPE *d;
  while ((d = READDIR (dp)) != NULL)
    {
      if (select != NULL)
        {
          int selected = (*select) (d);

	  /* The SELECT function might have set errno to non-zero on
	     success.  It was zero before and it needs to be again to
	     make the later tests work.  */
	  __set_errno (0);

          if (!selected)
            continue;
        }

      if (__glibc_unlikely (c.cnt == vsize))
        {
          if (vsize == 0)
            vsize = 10;
          else
            vsize *= 2;
          DIRENT_TYPE **new = realloc (v, vsize * sizeof *v);
          if (new == NULL)
            break;
          c.v = v = new;
        }

      size_t dsize = &d->d_name[_D_ALLOC_NAMLEN (d)] - (char *) d;
      DIRENT_TYPE *vnew = malloc (dsize);
      if (vnew == NULL)
        break;
      v[c.cnt++] = (DIRENT_TYPE *) memcpy (vnew, d, dsize);

      /* Ignore errors from readdir, malloc or realloc.  These functions
	 might have set errno to non-zero on success.  It was zero before
	 and it needs to be again to make the latter tests work.  */
      __set_errno (0);
    }

  if (__glibc_likely (errno == 0))
    {
      __closedir (dp);

      /* Sort the list if we have a comparison function to sort with.  */
      if (cmp != NULL)
	qsort (v, c.cnt, sizeof *v, (__compar_fn_t) cmp);

      *namelist = v;
      result = c.cnt;
    }
  else
    {
      /* This frees everything and calls closedir.  */
      __scandir_cancel_handler (&c);
      result = -1;
    }

  __libc_cleanup_pop (0);

  if (result >= 0)
    __set_errno (save);
  return result;
}
