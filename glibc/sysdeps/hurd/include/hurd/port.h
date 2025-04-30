#ifndef	_HURD_PORT_H
#include_next <hurd/port.h>

#ifndef _ISOMAC
#include <libc-lock.h>

struct _hurd_port_use_data
  {
     struct hurd_port *p;
     struct hurd_userlink link;
     mach_port_t port;
  };

extern void _hurd_port_use_cleanup (void *arg);

/* Like HURD_PORT_USE, but cleans fd on cancel.  */
#define	HURD_PORT_USE_CANCEL(portcell, expr)				      \
  ({ struct _hurd_port_use_data __d;					      \
     mach_port_t port;							      \
     __typeof(expr) __result;						      \
     void *__crit;							      \
     __d.p = (portcell);						      \
     __crit = _hurd_critical_section_lock ();				      \
     __d.port = port = _hurd_port_get (__d.p, &__d.link);		      \
     __libc_cleanup_push (_hurd_port_use_cleanup, &__d);		      \
     _hurd_critical_section_unlock (__crit);				      \
     __result = (expr);							      \
     __libc_cleanup_pop (1);						      \
     __result; })

libc_hidden_proto (_hurd_port_locked_get)
libc_hidden_proto (_hurd_port_locked_set)
#ifdef _HURD_PORT_H_HIDDEN_DEF
libc_hidden_def (_hurd_port_locked_get)
libc_hidden_def (_hurd_port_locked_set)
#endif
#endif
#endif
