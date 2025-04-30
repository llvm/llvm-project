#ifndef	_HURD_H
#include_next <hurd.h>

void _hurd_libc_proc_init (char **argv);

/* Like __USEPORT, but cleans fd on cancel.  */
#define	__USEPORT_CANCEL(which, expr) \
  HURD_PORT_USE_CANCEL (&_hurd_ports[INIT_PORT_##which], (expr))

#ifndef _ISOMAC
libc_hidden_proto (_hurd_exec_paths)
libc_hidden_proto (_hurd_init)
libc_hidden_proto (_hurd_libc_proc_init)
#endif
#endif
