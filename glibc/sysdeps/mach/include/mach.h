#ifndef	_MACH_H
#include_next <mach.h>
#include <mach-shortcuts-hidden.h>
#ifndef _ISOMAC
libc_hidden_proto (__mach_msg_destroy)
libc_hidden_proto (__mach_msg)
#endif
#endif
