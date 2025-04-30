#ifndef	_HURD_FD_H
#include_next <hurd/fd.h>

#ifndef _ISOMAC
#include <libc-lock.h>

struct _hurd_fd_port_use_data
  {
     struct hurd_fd *d;
     struct hurd_userlink ulink, ctty_ulink;
     io_t port, ctty;
  };

extern void _hurd_fd_port_use_cleanup (void *arg);

/* Like HURD_DPORT_USE, but cleans fd on cancel.  */
#define HURD_DPORT_USE_CANCEL(fd, expr) \
  HURD_FD_USE ((fd), HURD_FD_PORT_USE_CANCEL (descriptor, (expr)))

/* Like HURD_FD_PORT_USE, but cleans fd on cancel.  */
#define	HURD_FD_PORT_USE_CANCEL(fd, expr)				      \
  ({ error_t __result;							      \
     struct _hurd_fd_port_use_data __d;					      \
     io_t port, ctty;							      \
     void *__crit;							      \
     __d.d = (fd);							      \
     __crit = _hurd_critical_section_lock ();				      \
     __spin_lock (&__d.d->port.lock);					      \
     if (__d.d->port.port == MACH_PORT_NULL)				      \
       {								      \
	 __spin_unlock (&__d.d->port.lock);				      \
	 _hurd_critical_section_unlock (__crit);			      \
	 __result = EBADF;						      \
       }								      \
     else								      \
       {								      \
	 __d.ctty = ctty = _hurd_port_get (&__d.d->ctty, &__d.ctty_ulink);    \
	 __d.port = port = _hurd_port_locked_get (&__d.d->port, &__d.ulink);  \
	 __libc_cleanup_push (_hurd_fd_port_use_cleanup, &__d);		      \
	 _hurd_critical_section_unlock (__crit);			      \
	 __result = (expr);						      \
	 __libc_cleanup_pop (1);					      \
       }								      \
     __result; })

libc_hidden_proto (_hurd_intern_fd)
libc_hidden_proto (_hurd_fd_error)
libc_hidden_proto (_hurd_fd_error_signal)
#  ifdef _HURD_FD_H_HIDDEN_DEF
libc_hidden_def (_hurd_fd_error)
libc_hidden_def (_hurd_fd_error_signal)
#  endif
#endif
#endif
