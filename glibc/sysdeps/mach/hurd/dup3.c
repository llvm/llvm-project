/* Duplicate a file descriptor to a given number, with flags.  Hurd version.
   Copyright (C) 1991-2021 Free Software Foundation, Inc.

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
   License along with the GNU C Library; see the file COPYING.LIB.  If
   not, see <https://www.gnu.org/licenses/>.  */

#include <errno.h>
#include <fcntl.h>
#include <unistd.h>
#include <hurd.h>
#include <hurd/fd.h>

/* Duplicate FD to FD2, closing the old FD2 and making FD2 be
   open on the same file as FD is, and setting FD2's flags according to FLAGS.
   Return FD2 or -1.  */
int
__dup3 (int fd, int fd2, int flags)
{
  struct hurd_fd *d;

  /* Both passing flags different from O_CLOEXEC and FD2 being the same as FD
     are invalid.  */
  if ((flags & ~O_CLOEXEC
       || fd2 == fd)
      /* ... with the exception in case that dup2 behavior is requested: if FD
	 is valid and FD2 is already the same then just return it.  */
      && ! (flags == -1
	    && fd2 == fd))
    return __hurd_fail (EINVAL);

  /* Extract the ports and flags from FD.  */
  d = _hurd_fd_get (fd);
  if (d == NULL)
    return __hurd_fail (EBADF);

  HURD_CRITICAL_BEGIN;

  __spin_lock (&d->port.lock);
  if (d->port.port == MACH_PORT_NULL)
    {
      __spin_unlock (&d->port.lock);
      fd2 = __hurd_fail (EBADF);
    }
  else if (fd2 == fd)
    __spin_unlock (&d->port.lock);
  else
    {
      struct hurd_userlink ulink, ctty_ulink;
      int d_flags = d->flags;
      io_t ctty = _hurd_port_get (&d->ctty, &ctty_ulink);
      io_t port = _hurd_port_locked_get (&d->port, &ulink); /* Unlocks D.  */

      if (fd2 < 0)
	fd2 = __hurd_fail (EBADF);
      else
	{
	  /* Get a hold of the destination descriptor.  */
	  struct hurd_fd *d2;

	  __mutex_lock (&_hurd_dtable_lock);

	  if (fd2 >= _hurd_dtablesize)
	    {
	      /* The table is not large enough to hold the destination
		 descriptor.  Enlarge it as necessary to allocate this
		 descriptor.  */
	      __mutex_unlock (&_hurd_dtable_lock);
	      d2 = _hurd_alloc_fd (NULL, fd2);
	      if (d2)
		__spin_unlock (&d2->port.lock);
	      __mutex_lock (&_hurd_dtable_lock);
	    }
	  else
	    {
	      d2 = _hurd_dtable[fd2];
	      if (d2 == NULL)
		{
		  /* Must allocate a new one.  We don't initialize the port
		     cells with this call so that if it fails (out of
		     memory), we will not have already added user
		     references for the ports, which we would then have to
		     deallocate.  */
		  d2 = _hurd_dtable[fd2] = _hurd_new_fd (MACH_PORT_NULL,
							 MACH_PORT_NULL);
		}
	    }
	  __mutex_unlock (&_hurd_dtable_lock);

	  if (d2 == NULL)
	    {
	      fd2 = -1;
	      if (errno == EINVAL)
		errno = EBADF;	/* POSIX.1-1990 6.2.1.2 ll 54-55.  */
	    }
	  else
	    {
	      /* Give the ports each a user ref for the new descriptor.  */
	      __mach_port_mod_refs (__mach_task_self (), port,
				    MACH_PORT_RIGHT_SEND, 1);
	      if (ctty != MACH_PORT_NULL)
		__mach_port_mod_refs (__mach_task_self (), ctty,
				      MACH_PORT_RIGHT_SEND, 1);

	      /* Install the ports and flags in the new descriptor slot.  */
	      __spin_lock (&d2->port.lock);
	      if (flags & O_CLOEXEC)
		d2->flags = d_flags | FD_CLOEXEC;
	      else
		/* dup clears FD_CLOEXEC.  */
		d2->flags = d_flags & ~FD_CLOEXEC;
	      _hurd_port_set (&d2->ctty, ctty);
	      _hurd_port_locked_set (&d2->port, port); /* Unlocks D2.  */
	    }
	}

      _hurd_port_free (&d->port, &ulink, port);
      if (ctty != MACH_PORT_NULL)
	_hurd_port_free (&d->ctty, &ctty_ulink, port);
    }

  HURD_CRITICAL_END;

  return fd2;
}
libc_hidden_def (__dup3)
weak_alias (__dup3, dup3)
