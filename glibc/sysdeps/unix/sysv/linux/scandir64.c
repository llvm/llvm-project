/* Copyright (C) 2000-2021 Free Software Foundation, Inc.
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

#define scandir __no_scandir_decl
#include <dirent.h>
#undef scandir

int
__scandir64 (const char *dir, struct dirent64 ***namelist,
	   int (*select) (const struct dirent64 *),
	   int (*cmp) (const struct dirent64 **, const struct dirent64 **))
{
  return __scandir64_tail (__opendir (dir), namelist, select, cmp);
}

#if _DIRENT_MATCHES_DIRENT64
weak_alias (__scandir64, scandir64)
weak_alias (__scandir64, scandir)
#else
# include <shlib-compat.h>
versioned_symbol (libc, __scandir64, scandir64, GLIBC_2_2);
# if SHLIB_COMPAT (libc, GLIBC_2_1, GLIBC_2_2)
#  include <string.h>
#  include <errno.h>
#  include "olddirent.h"

int
__old_scandir64 (const char *dir, struct __old_dirent64 ***namelist,
		 int (*select) (const struct __old_dirent64 *),
		 int (*cmp) (const struct __old_dirent64 **,
			     const struct __old_dirent64 **))
{
  DIR *dp = __opendir (dir);
  struct __old_dirent64 **v = NULL;
  size_t vsize = 0;
  struct scandir_cancel_struct c;
  struct __old_dirent64 *d;
  int save;

  if (dp == NULL)
    return -1;

  save = errno;
  __set_errno (0);

  c.dp = dp;
  c.v = NULL;
  c.cnt = 0;
  __libc_cleanup_push (__scandir_cancel_handler, &c);

  while ((d = __old_readdir64 (dp)) != NULL)
    {
      int use_it = select == NULL;

      if (! use_it)
	{
	  use_it = select (d);
	  /* The select function might have changed errno.  It was
	     zero before and it need to be again to make the latter
	     tests work.  */
	  __set_errno (0);
	}

      if (use_it)
	{
	  struct __old_dirent64 *vnew;
	  size_t dsize;

	  /* Ignore errors from select or readdir */
	  __set_errno (0);

	  if (__glibc_unlikely (c.cnt == vsize))
	    {
	      struct __old_dirent64 **new;
	      if (vsize == 0)
		vsize = 10;
	      else
		vsize *= 2;
	      new = (struct __old_dirent64 **) realloc (v,
							vsize * sizeof (*v));
	      if (new == NULL)
		break;
	      v = new;
	      c.v = (void *) v;
	    }

	  dsize = &d->d_name[_D_ALLOC_NAMLEN (d)] - (char *) d;
	  vnew = (struct __old_dirent64 *) malloc (dsize);
	  if (vnew == NULL)
	    break;

	  v[c.cnt++] = (struct __old_dirent64 *) memcpy (vnew, d, dsize);
	}
    }

  if (__builtin_expect (errno, 0) != 0)
    {
      save = errno;

      while (c.cnt > 0)
	free (v[--c.cnt]);
      free (v);
      c.cnt = -1;
    }
  else
    {
      /* Sort the list if we have a comparison function to sort with.  */
      if (cmp != NULL)
	qsort (v, c.cnt, sizeof (*v),
	       (int (*) (const void *, const void *)) cmp);

      *namelist = v;
    }

  __libc_cleanup_pop (0);

  (void) __closedir (dp);
  __set_errno (save);

  return c.cnt;
}
compat_symbol (libc, __old_scandir64, scandir64, GLIBC_2_1);

# endif /* SHLIB_COMPAT (libc, GLIBC_2_1, GLIBC_2_2)  */
#endif /* _DIRENT_MATCHES_DIRENT64  */
