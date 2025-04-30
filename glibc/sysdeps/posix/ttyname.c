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

#include <errno.h>
#include <limits.h>
#include <stddef.h>
#include <dirent.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <string.h>
#include <stdlib.h>

char *__ttyname;

static char *getttyname (int fd, dev_t mydev, ino_t myino,
			 int save, int *dostat);


libc_freeres_ptr (static char *getttyname_name);

static char *
getttyname (int fd, dev_t mydev, ino_t myino, int save, int *dostat)
{
  static const char dev[] = "/dev";
  static size_t namelen;
  struct stat st;
  DIR *dirstream;
  struct dirent *d;

  dirstream = __opendir (dev);
  if (dirstream == NULL)
    {
      *dostat = -1;
      return NULL;
    }

  while ((d = __readdir (dirstream)) != NULL)
    if (((ino_t) d->d_fileno == myino || *dostat)
	&& strcmp (d->d_name, "stdin")
	&& strcmp (d->d_name, "stdout")
	&& strcmp (d->d_name, "stderr"))
      {
	size_t dlen = _D_ALLOC_NAMLEN (d);
	if (sizeof (dev) + dlen > namelen)
	  {
	    free (getttyname_name);
	    namelen = 2 * (sizeof (dev) + dlen); /* Big enough.  */
	    getttyname_name = malloc (namelen);
	    if (! getttyname_name)
	      {
		*dostat = -1;
		/* Perhaps it helps to free the directory stream buffer.  */
		(void) __closedir (dirstream);
		return NULL;
	      }
	    *((char *) __mempcpy (getttyname_name, dev, sizeof (dev) - 1))
	      = '/';
	  }
	(void) __mempcpy (&getttyname_name[sizeof (dev)], d->d_name, dlen);
	if (stat (getttyname_name, &st) == 0
#ifdef _STATBUF_ST_RDEV
	    && S_ISCHR (st.st_mode) && st.st_rdev == mydev
#else
	    && (ino_t) d->d_fileno == myino && st.st_dev == mydev
#endif
	   )
	  {
	    (void) __closedir (dirstream);
	    __ttyname = getttyname_name;
	    __set_errno (save);
	    return getttyname_name;
	  }
      }

  (void) __closedir (dirstream);
  __set_errno (save);
  return NULL;
}

/* Return the pathname of the terminal FD is open on, or NULL on errors.
   The returned storage is good only until the next call to this function.  */
char *
ttyname (int fd)
{
  struct stat st;
  int dostat = 0;
  char *name;
  int save = errno;

  if (!__isatty (fd))
    return NULL;

  if (fstat (fd, &st) < 0)
    return NULL;

#ifdef _STATBUF_ST_RDEV
  name = getttyname (fd, st.st_rdev, st.st_ino, save, &dostat);
#else
  name = getttyname (fd, st.st_dev, st.st_ino, save, &dostat);
#endif

  if (!name && dostat != -1)
    {
      dostat = 1;
#ifdef _STATBUF_ST_RDEV
      name = getttyname (fd, st.st_rdev, st.st_ino, save, &dostat);
#else
      name = getttyname (fd, st.st_dev, st.st_ino, save, &dostat);
#endif
    }

  return name;
}
