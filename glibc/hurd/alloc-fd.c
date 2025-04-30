/* Copyright (C) 1994-2021 Free Software Foundation, Inc.
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

#include <hurd.h>
#include <hurd/fd.h>
#include <hurd/resource.h>
#include <stdlib.h>
#include "hurdmalloc.h"		/* XXX */

/* Allocate a new file descriptor and return it, locked.  The new
   descriptor number will be no less than FIRST_FD.  If the table is full,
   set errno to EMFILE and return NULL.  If FIRST_FD is negative or bigger
   than the size of the table, set errno to EINVAL and return NULL.  */

struct hurd_fd *
_hurd_alloc_fd (int *fd, int first_fd)
{
  int i;
  void *crit;
  long int rlimit;

  if (first_fd < 0)
    {
      errno = EINVAL;
      return NULL;
    }

  crit = _hurd_critical_section_lock ();

  __mutex_lock (&_hurd_dtable_lock);

 search:
  for (i = first_fd; i < _hurd_dtablesize; ++i)
    {
      struct hurd_fd *d = _hurd_dtable[i];
      if (d == NULL)
	{
	  /* Allocate a new descriptor structure for this slot,
	     initializing its port cells to nil.  The test below will catch
	     and return this descriptor cell after locking it.  */
	  d = _hurd_new_fd (MACH_PORT_NULL, MACH_PORT_NULL);
	  if (d == NULL)
	    {
	      __mutex_unlock (&_hurd_dtable_lock);
	      _hurd_critical_section_unlock (crit);
	      return NULL;
	    }
	  _hurd_dtable[i] = d;
	}

      __spin_lock (&d->port.lock);
      if (d->port.port == MACH_PORT_NULL)
	{
	  __mutex_unlock (&_hurd_dtable_lock);
	  _hurd_critical_section_unlock (crit);
	  if (fd != NULL)
	    *fd = i;
	  return d;
	}
      else
	__spin_unlock (&d->port.lock);
    }

  __mutex_lock (&_hurd_rlimit_lock);
  rlimit = _hurd_rlimits[RLIMIT_OFILE].rlim_cur;
  __mutex_unlock (&_hurd_rlimit_lock);

  if (first_fd < rlimit)
    {
      /* The descriptor table is full.  Check if we have reached the
	 resource limit, or only the allocated size.  */
      if (_hurd_dtablesize < rlimit)
	{
	  /* Enlarge the table.  */
	  int save = errno;
	  struct hurd_fd **new;
	  /* Try to double the table size, but don't exceed the limit,
	     and make sure it exceeds FIRST_FD.  */
	  int size = _hurd_dtablesize * 2;
	  if (size > rlimit)
	    size = rlimit;
	  else if (size <= first_fd)
	    size = first_fd + 1;

	  if (size * sizeof (*_hurd_dtable) < size)
	    {
	      /* Integer overflow! */
	      errno = ENOMEM;
	      goto out;
	    }

	  /* If we fail to allocate that, decrement the desired size
	     until we succeed in allocating it.  */
	  do
	    new = realloc (_hurd_dtable, size * sizeof (*_hurd_dtable));
	  while (new == NULL && size-- > first_fd);

	  if (new != NULL)
	    {
	      /* We managed to allocate a new table.  Now install it.  */
	      errno = save;
	      if (first_fd < _hurd_dtablesize)
		first_fd = _hurd_dtablesize;
	      /* Initialize the new slots.  */
	      for (i = _hurd_dtablesize; i < size; ++i)
		new[i] = NULL;
	      _hurd_dtablesize = size;
	      _hurd_dtable = new;
	      /* Go back to the loop to initialize the first new slot.  */
	      goto search;
	    }
	  else
	    errno = ENOMEM;
	}
      else
	errno = EMFILE;
    }
  else
    errno = EINVAL;		/* Bogus FIRST_FD value.  */

 out:
  __mutex_unlock (&_hurd_dtable_lock);
  _hurd_critical_section_unlock (crit);

  return NULL;
}
