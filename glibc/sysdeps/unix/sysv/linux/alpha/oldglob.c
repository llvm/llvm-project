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
   License along with the GNU C Library.  If not, see
   <https://www.gnu.org/licenses/>.  */

/* This file contains only wrappers around the real glob functions.  It
   became necessary since the glob_t structure changed.  */
#include <sys/types.h>
#include <glob.h>
#include <shlib-compat.h>

#if SHLIB_COMPAT (libc, GLIBC_2_0, GLIBC_2_1)

/* This is the old structure.  The difference is that the gl_pathc and
   gl_offs elements have type `int'.  */
typedef struct
  {
    int gl_pathc;		/* Count of paths matched by the pattern.  */
    char **gl_pathv;		/* List of matched pathnames.  */
    int gl_offs;		/* Slots to reserve in `gl_pathv'.  */
    int gl_flags;		/* Set to FLAGS, maybe | GLOB_MAGCHAR.  */

    /* If the GLOB_ALTDIRFUNC flag is set, the following functions
       are used instead of the normal file access functions.  */
    void (*gl_closedir) (void *);
    struct dirent *(*gl_readdir) (void *);
    void *(*gl_opendir) (const char *);
    int (*gl_lstat) (const char *, struct stat *);
    int (*gl_stat) (const char *, struct stat *);
  } old_glob_t;


int
attribute_compat_text_section
__old_glob (const char *pattern, int flags,
	    int (*errfunc) (const char *, int),
	    old_glob_t *pglob)
{
  glob_t correct;
  int result;

  /* Construct an object of correct type.  */
  correct.gl_pathc = pglob->gl_pathc;
  correct.gl_pathv = pglob->gl_pathv;
  correct.gl_offs = pglob->gl_offs;
  correct.gl_flags = pglob->gl_flags;
  correct.gl_closedir = pglob->gl_closedir;
  correct.gl_readdir = pglob->gl_readdir;
  correct.gl_opendir = pglob->gl_opendir;
  /* Set gl_lstat and gl_stat for both gl_stat for compatibility with old
     implementation that did not follow dangling symlinks.  */
  correct.gl_lstat = pglob->gl_stat;
  correct.gl_stat = pglob->gl_stat;

  result = glob (pattern, flags, errfunc, &correct);

  /* And convert it back.  */
  pglob->gl_pathc = correct.gl_pathc;
  pglob->gl_pathv = correct.gl_pathv;
  pglob->gl_offs = correct.gl_offs;
  pglob->gl_flags = correct.gl_flags;
  pglob->gl_closedir = correct.gl_closedir;
  pglob->gl_readdir = correct.gl_readdir;
  pglob->gl_opendir = correct.gl_opendir;
  /* Only need to restore gl_stat.  */
  pglob->gl_stat = correct.gl_stat;

  return result;
}
compat_symbol (libc, __old_glob, glob, GLIBC_2_0);


/* Free storage allocated in PGLOB by a previous `glob' call.  */
void
attribute_compat_text_section
__old_globfree (old_glob_t *pglob)
{
  glob_t correct;

  /* We only need these two symbols.  */
  correct.gl_pathc = pglob->gl_pathc;
  correct.gl_pathv = pglob->gl_pathv;
  correct.gl_offs = pglob->gl_offs;

  globfree (&correct);
}
compat_symbol (libc, __old_globfree, globfree, GLIBC_2_0);

#endif
