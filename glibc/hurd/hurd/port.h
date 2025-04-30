/* Lightweight user references for ports.
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

#ifndef	_HURD_PORT_H

#define	_HURD_PORT_H	1
#include <features.h>

#include <mach.h>
#include <hurd/userlink.h>
#include <spin-lock.h>


/* Structure describing a cell containing a port.  With the lock held, a
   user extracts PORT, and attaches his own link (in local storage) to the
   USERS chain.  PORT can then safely be used.  When PORT is no longer
   needed, with the lock held, the user removes his link from the chain.
   If his link is the last, and PORT has changed since he fetched it, the
   user deallocates the port he used.  See <hurd/userlink.h>.  */

struct hurd_port
  {
    spin_lock_t lock;		/* Locks rest.  */
    struct hurd_userlink *users; /* Chain of users; see below.  */
    mach_port_t port;		/* Port. */
  };


/* Evaluate EXPR with the variable `port' bound to the port in PORTCELL.  */
/* Also see HURD_PORT_USE_CANCEL.  */

#define	HURD_PORT_USE(portcell, expr)					      \
  ({ struct hurd_port *const __p = (portcell);				      \
     struct hurd_userlink __link;					      \
     const mach_port_t port = _hurd_port_get (__p, &__link);		      \
     __typeof(expr) __result = (expr);					      \
     _hurd_port_free (__p, &__link, port);				      \
     __result; })


#ifndef _HURD_PORT_H_EXTERN_INLINE
#define _HURD_PORT_H_EXTERN_INLINE __extern_inline
#endif


/* Initialize *PORT to INIT.  */

extern void _hurd_port_init (struct hurd_port *port, mach_port_t init);

#if defined __USE_EXTERN_INLINES && defined _LIBC
# if IS_IN (libc)
_HURD_PORT_H_EXTERN_INLINE void
_hurd_port_init (struct hurd_port *port, mach_port_t init)
{
  __spin_lock_init (&port->lock);
  port->users = NULL;
  port->port = init;
}
# endif
#endif


/* Cleanup function for non-local exits.  */
extern void _hurd_port_cleanup (void *, jmp_buf, int);

/* Get a reference to *PORT, which is locked.
   Pass return value and LINK to _hurd_port_free when done.  */

extern mach_port_t
_hurd_port_locked_get (struct hurd_port *port,
		       struct hurd_userlink *link);

#if defined __USE_EXTERN_INLINES && defined _LIBC
# if IS_IN (libc)
_HURD_PORT_H_EXTERN_INLINE mach_port_t
_hurd_port_locked_get (struct hurd_port *port,
		       struct hurd_userlink *link)
{
  mach_port_t result;
  result = port->port;
  if (result != MACH_PORT_NULL)
    {
      link->cleanup = &_hurd_port_cleanup;
      link->cleanup_data = (void *) result;
      _hurd_userlink_link (&port->users, link);
    }
  __spin_unlock (&port->lock);
  return result;
}
# endif
#endif

/* Same, but locks PORT first.  */

extern mach_port_t
_hurd_port_get (struct hurd_port *port,
		struct hurd_userlink *link);

#if defined __USE_EXTERN_INLINES && defined _LIBC
# if IS_IN (libc)
_HURD_PORT_H_EXTERN_INLINE mach_port_t
_hurd_port_get (struct hurd_port *port,
		struct hurd_userlink *link)
{
  mach_port_t result;
  HURD_CRITICAL_BEGIN;
  __spin_lock (&port->lock);
  result = _hurd_port_locked_get (port, link);
  HURD_CRITICAL_END;
  return result;
}
# endif
#endif


/* Relocate LINK to NEW_LINK.
   To be used when e.g. reallocating a link array.  */

extern void
_hurd_port_move (struct hurd_port *port,
		 struct hurd_userlink *new_link,
		 struct hurd_userlink *link);

#if defined __USE_EXTERN_INLINES && defined _LIBC
# if IS_IN (libc)
_HURD_PORT_H_EXTERN_INLINE void
_hurd_port_move (struct hurd_port *port,
		 struct hurd_userlink *new_link,
		 struct hurd_userlink *link)
{
  HURD_CRITICAL_BEGIN;
  __spin_lock (&port->lock);
  _hurd_userlink_move (new_link, link);
  __spin_unlock (&port->lock);
  HURD_CRITICAL_END;
}
# endif
#endif


/* Free a reference gotten with `USED_PORT = _hurd_port_get (PORT, LINK);' */

extern void
_hurd_port_free (struct hurd_port *port,
		 struct hurd_userlink *link,
		 mach_port_t used_port);

#if defined __USE_EXTERN_INLINES && defined _LIBC
# if IS_IN (libc)
_HURD_PORT_H_EXTERN_INLINE void
_hurd_port_free (struct hurd_port *port,
		 struct hurd_userlink *link,
		 mach_port_t used_port)
{
  int dealloc;
  if (used_port == MACH_PORT_NULL)
    /* When we fetch an empty port cell with _hurd_port_get,
       it does not link us on the users chain, since there is
       no shared resource.  */
    return;
  HURD_CRITICAL_BEGIN;
  __spin_lock (&port->lock);
  dealloc = _hurd_userlink_unlink (link);
  __spin_unlock (&port->lock);
  HURD_CRITICAL_END;
  if (dealloc)
    __mach_port_deallocate (__mach_task_self (), used_port);
}
# endif
#endif


/* Set *PORT's port to NEWPORT.  NEWPORT's reference is consumed by PORT->port.
   PORT->lock is locked.  */

extern void _hurd_port_locked_set (struct hurd_port *port, mach_port_t newport);

#if defined __USE_EXTERN_INLINES && defined _LIBC
# if IS_IN (libc)
_HURD_PORT_H_EXTERN_INLINE void
_hurd_port_locked_set (struct hurd_port *port, mach_port_t newport)
{
  mach_port_t old;
  old = _hurd_userlink_clear (&port->users) ? port->port : MACH_PORT_NULL;
  port->port = newport;
  __spin_unlock (&port->lock);
  if (old != MACH_PORT_NULL)
    __mach_port_deallocate (__mach_task_self (), old);
}
# endif
#endif

/* Same, but locks PORT first.  */

extern void _hurd_port_set (struct hurd_port *port, mach_port_t newport);

#if defined __USE_EXTERN_INLINES && defined _LIBC
# if IS_IN (libc)
_HURD_PORT_H_EXTERN_INLINE void
_hurd_port_set (struct hurd_port *port, mach_port_t newport)
{
  HURD_CRITICAL_BEGIN;
  __spin_lock (&port->lock);
  _hurd_port_locked_set (port, newport);
  HURD_CRITICAL_END;
}
# endif
#endif


#endif	/* hurd/port.h */
