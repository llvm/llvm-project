/* Determine current working directory.  Linux version.
   Copyright (C) 1997-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Ulrich Drepper <drepper@cygnus.com>, 1997.

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

#include <assert.h>
#include <errno.h>
#include <limits.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/param.h>

#include <sysdep.h>
#include <sys/syscall.h>


/* If we compile the file for use in ld.so we don't need the feature
   that getcwd() allocates the buffers itself.  */
#if IS_IN (rtld)
# define NO_ALLOCATION	1
#endif


/* The "proc" filesystem provides an easy method to retrieve the value.
   For each process, the corresponding directory contains a symbolic link
   named `cwd'.  Reading the content of this link immediate gives us the
   information.  But we have to take care for systems which do not have
   the proc filesystem mounted.  Use the POSIX implementation in this case.  */

/* Get the code for the generic version.  */
#define GETCWD_RETURN_TYPE	static char *
#include <sysdeps/posix/getcwd.c>

char *
__getcwd (char *buf, size_t size)
{
  char *path;
  char *result;

#ifndef NO_ALLOCATION
  size_t alloc_size = size;
  if (size == 0)
    {
      if (buf != NULL)
	{
	  __set_errno (EINVAL);
	  return NULL;
	}

      alloc_size = MAX (PATH_MAX, __getpagesize ());
    }

  if (buf == NULL)
    {
      path = malloc (alloc_size);
      if (path == NULL)
	return NULL;
    }
  else
#else
# define alloc_size size
#endif
    path = buf;

  int retval;

  retval = INLINE_SYSCALL (getcwd, 2, path, alloc_size);
  if (retval > 0 && path[0] == '/')
    {
#ifndef NO_ALLOCATION
      if (buf == NULL && size == 0)
	/* Ensure that the buffer is only as large as necessary.  */
	buf = realloc (path, (size_t) retval);

      if (buf == NULL)
	/* Either buf was NULL all along, or `realloc' failed but
	   we still have the original string.  */
	buf = path;
#endif

      return buf;
    }

  /* The system call either cannot handle paths longer than a page
     or can succeed without returning an absolute path.  Just use the
     generic implementation right away.  */
  if (retval >= 0 || errno == ENAMETOOLONG)
    {
#ifndef NO_ALLOCATION
      if (buf == NULL && size == 0)
	{
	  free (path);
	  path = NULL;
	}
#endif

      result = __getcwd_generic (path, size);

#ifndef NO_ALLOCATION
      if (result == NULL && buf == NULL && size != 0)
	free (path);
#endif

      return result;
    }

  /* It should never happen that the `getcwd' syscall failed because
     the buffer is too small if we allocated the buffer ourselves
     large enough.  */
  assert (errno != ERANGE || buf != NULL || size != 0);

#ifndef NO_ALLOCATION
  if (buf == NULL)
    free (path);
#endif

  return NULL;
}
libc_hidden_def (__getcwd)
weak_alias (__getcwd, getcwd)
