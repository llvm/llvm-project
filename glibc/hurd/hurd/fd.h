/* File descriptors.
   Copyright (C) 1993-2021 Free Software Foundation, Inc.
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

#ifndef	_HURD_FD_H

#define	_HURD_FD_H	1
#include <features.h>

#include <hurd/hurd_types.h>
#include <hurd/port.h>
#include <sys/socket.h>
#include <sys/select.h>
#include <fcntl.h>
#include <bits/types/sigset_t.h>


/* Structure representing a file descriptor.  */

struct hurd_fd
  {
    struct hurd_port port;	/* io server port.  */
    int flags;			/* fcntl flags; locked by port.lock.  */

    /* Normal port to the ctty.  When `port' is our ctty, this is a port to
       the same io object but which never returns EBACKGROUND; when not,
       this is nil.  */
    struct hurd_port ctty;
  };


/* Current file descriptor table.  */

#if defined __USE_EXTERN_INLINES && defined _LIBC
#include <lock-intern.h>
extern int _hurd_dtablesize;
extern struct hurd_fd **_hurd_dtable;
extern struct mutex _hurd_dtable_lock; /* Locks those two variables.  */
#endif

#include <hurd/signal.h>

#ifndef _HURD_FD_H_EXTERN_INLINE
#define _HURD_FD_H_EXTERN_INLINE __extern_inline
#endif

/* Returns the descriptor cell for FD.  If FD is invalid or unused, return
   NULL.  The cell is unlocked; when ready to use it, lock it and check for
   it being unused.  */

extern struct hurd_fd *_hurd_fd_get (int fd);

#if defined __USE_EXTERN_INLINES && defined _LIBC
# if IS_IN (libc)
_HURD_FD_H_EXTERN_INLINE struct hurd_fd *
_hurd_fd_get (int fd)
{
  struct hurd_fd *descriptor;

  HURD_CRITICAL_BEGIN;
  __mutex_lock (&_hurd_dtable_lock);
  if (fd < 0 || fd >= _hurd_dtablesize)
    descriptor = NULL;
  else
    {
      struct hurd_fd *cell = _hurd_dtable[fd];
      if (cell == NULL)
	/* No descriptor allocated at this index.  */
	descriptor = NULL;
      else
	{
	  __spin_lock (&cell->port.lock);
	  if (cell->port.port == MACH_PORT_NULL)
	    /* The descriptor at this index has no port in it.
	       This happens if it existed before but was closed.  */
	    descriptor = NULL;
	  else
	    descriptor = cell;
	  __spin_unlock (&cell->port.lock);
	}
    }
  __mutex_unlock (&_hurd_dtable_lock);
  HURD_CRITICAL_END;

  return descriptor;
}
# endif
#endif


/* Evaluate EXPR with the variable `descriptor' bound to a pointer to the
   file descriptor structure for FD.   */

#define	HURD_FD_USE(fd, expr)						      \
  ({ struct hurd_fd *descriptor = _hurd_fd_get (fd);			      \
     descriptor == NULL ? EBADF : (expr); })

/* Evaluate EXPR with the variable `port' bound to the port to FD, and
   `ctty' bound to the ctty port.  */

#define HURD_DPORT_USE(fd, expr) \
  HURD_FD_USE ((fd), HURD_FD_PORT_USE (descriptor, (expr)))

/* Likewise, but FD is a pointer to the file descriptor structure.  */
/* Also see HURD_FD_PORT_USE_CANCEL.  */

#define	HURD_FD_PORT_USE(fd, expr)					      \
  ({ error_t __result;							      \
     struct hurd_fd *const __d = (fd);					      \
     struct hurd_userlink __ulink, __ctty_ulink;			      \
     io_t port, ctty;							      \
     void *crit = _hurd_critical_section_lock ();			      \
     __spin_lock (&__d->port.lock);					      \
     if (__d->port.port == MACH_PORT_NULL)				      \
       {								      \
	 __spin_unlock (&__d->port.lock);				      \
	 _hurd_critical_section_unlock (crit);				      \
	 __result = EBADF;						      \
       }								      \
     else								      \
       {								      \
	 ctty = _hurd_port_get (&__d->ctty, &__ctty_ulink);		      \
	 port = _hurd_port_locked_get (&__d->port, &__ulink);		      \
	 _hurd_critical_section_unlock (crit);				      \
	 __result = (expr);						      \
	 _hurd_port_free (&__d->port, &__ulink, port);			      \
	 if (ctty != MACH_PORT_NULL)					      \
	   _hurd_port_free (&__d->ctty, &__ctty_ulink, ctty);		      \
       }								      \
     __result; })

#include <errno.h>
#include <bits/types/error_t.h>

/* Check if ERR should generate a signal.
   Returns the signal to take, or zero if none.  */

extern int _hurd_fd_error_signal (error_t err);

#ifdef __USE_EXTERN_INLINES
_HURD_FD_H_EXTERN_INLINE int
_hurd_fd_error_signal (error_t err)
{
  switch (err)
    {
    case EMACH_SEND_INVALID_DEST:
    case EMIG_SERVER_DIED:
      /* The server has disappeared!  */
      return SIGLOST;
    case EPIPE:
      return SIGPIPE;
    default:
      /* Having a default case avoids -Wenum-switch warnings.  */
      return 0;
    }
}
#endif

/* Handle an error from an RPC on a file descriptor's port.  You should
   always use this function to handle errors from RPCs made on file
   descriptor ports.  Some errors are translated into signals.  */

extern error_t _hurd_fd_error (int fd, error_t err);

#ifdef __USE_EXTERN_INLINES
_HURD_FD_H_EXTERN_INLINE error_t
_hurd_fd_error (int fd, error_t err)
{
  int signo = _hurd_fd_error_signal (err);
  if (signo)
    {
      const struct hurd_signal_detail detail
	= { exc: 0, exc_code: 0, exc_subcode: 0, code: fd, error: err };
      _hurd_raise_signal (NULL, signo, &detail);
    }
  return err;
}
#endif

/* Handle error code ERR from an RPC on file descriptor FD's port.
   Set `errno' to the appropriate error code, and always return -1.  */

extern int __hurd_dfail (int fd, error_t err);

#ifdef __USE_EXTERN_INLINES
_HURD_FD_H_EXTERN_INLINE int
__hurd_dfail (int fd, error_t err)
{
  errno = _hurd_fd_error (fd, err);
  return -1;
}
#endif

/* Likewise, but do not raise SIGPIPE on EPIPE if flags contain
   MSG_NOSIGNAL.  */

extern int __hurd_sockfail (int fd, int flags, error_t err);

#ifdef __USE_EXTERN_INLINES
_HURD_FD_H_EXTERN_INLINE int
__hurd_sockfail (int fd, int flags, error_t err)
{
  if (!(flags & MSG_NOSIGNAL) || err != EPIPE)
    err = _hurd_fd_error (fd, err);
  errno = err;
  return -1;
}
#endif

/* Set up *FD to have PORT its server port, doing appropriate ctty magic.
   Does no locking or unlocking.  */

extern void _hurd_port2fd (struct hurd_fd *fd, io_t port, int flags);

/* Allocate a new file descriptor and install PORT in it (doing any
   appropriate ctty magic); consumes a user reference on PORT.  FLAGS are
   as for `open'; only O_IGNORE_CTTY and O_CLOEXEC are meaningful, but all are
   saved.

   If the descriptor table is full, set errno, and return -1.
   If DEALLOC is nonzero, deallocate PORT first.  */

extern int _hurd_intern_fd (io_t port, int flags, int dealloc);

/* Allocate a new file descriptor in the table and return it, locked.  The
   new descriptor number will be no less than FIRST_FD.  If the table is
   full, set errno to EMFILE and return NULL.  If FIRST_FD is negative or
   bigger than the size of the table, set errno to EINVAL and return NULL.  */

extern struct hurd_fd *_hurd_alloc_fd (int *fd_ptr, int first_fd);

/* Allocate a new file descriptor structure and initialize its port cells
   with PORT and CTTY.  (This does not affect the descriptor table.)  */

extern struct hurd_fd *_hurd_new_fd (io_t port, io_t ctty);

/* Close a file descriptor, making it available for future reallocation.  */

extern error_t _hurd_fd_close (struct hurd_fd *fd);

/* Read and write data from a file descriptor; just like `read' and `write'
   if OFFSET is -1, or like `pread' and `pwrite' if OFFSET is not -1.
   If successful, stores the amount actually read or written in *NBYTES.  */

extern error_t _hurd_fd_read (struct hurd_fd *fd,
			      void *buf, size_t *nbytes, __loff_t offset);
extern error_t _hurd_fd_write (struct hurd_fd *fd,
			       const void *buf, size_t *nbytes, __loff_t offset);


/* Call *RPC on PORT and/or CTTY; if a call on CTTY returns EBACKGROUND,
   generate SIGTTIN/SIGTTOU or EIO as appropriate.  */

extern error_t _hurd_ctty_input (io_t port, io_t ctty, error_t (*rpc) (io_t));
extern error_t _hurd_ctty_output (io_t port, io_t ctty, error_t (*rpc) (io_t));


/* The guts of `select' and `poll'.  Check the first NFDS descriptors
   either in POLLFDS (if nonnull) or in each of READFDS, WRITEFDS,
   EXCEPTFDS that is nonnull.  If TIMEOUT is not NULL, time out after
   waiting the interval specified therein.  If SIGMASK is nonnull,
   the set of blocked signals is temporarily set to that during this call.
   Returns the number of ready descriptors, or -1 for errors.  */
struct pollfd;
struct timespec;
extern int _hurd_select (int nfds, struct pollfd *pollfds,
			 fd_set *readfds, fd_set *writefds, fd_set *exceptfds,
			 const struct timespec *timeout,
			 const sigset_t *sigmask);

/* Apply AT_FLAGS on FLAGS, in preparation for calling
   __hurd_file_name_lookup.  */

#if defined __USE_EXTERN_INLINES && defined _LIBC
# if IS_IN (libc)
_HURD_FD_H_EXTERN_INLINE error_t
__hurd_at_flags (int *at_flags, int *flags)
{
  if ((*at_flags & AT_SYMLINK_FOLLOW) && (*at_flags & AT_SYMLINK_NOFOLLOW))
    return EINVAL;

  *flags |= (*at_flags & AT_SYMLINK_NOFOLLOW) ? O_NOLINK : 0;
  *at_flags &= ~AT_SYMLINK_NOFOLLOW;
  if (*at_flags & AT_SYMLINK_FOLLOW)
    *flags &= ~O_NOLINK;
  *at_flags &= ~AT_SYMLINK_FOLLOW;
  if (*at_flags != 0)
    return EINVAL;

  return 0;
}
# endif
#endif

/* Variant of file_name_lookup used in *at function implementations.
   AT_FLAGS may only contain AT_SYMLINK_FOLLOW or AT_SYMLINK_NOFOLLOW,
   which will remove and add O_NOLINK from FLAGS respectively.
   Other bits cause EINVAL.  */
extern file_t __file_name_lookup_at (int fd, int at_flags,
				     const char *file_name,
				     int flags, mode_t mode);

/* Variant of file_name_split used in *at function implementations.  */
extern file_t __file_name_split_at (int fd, const char *file_name,
				    char **name);

/* Variant of directory_name_split used in *at function implementations.  */
extern file_t __directory_name_split_at (int fd, const char *directory_name,
					 char **name);



#endif	/* hurd/fd.h */
