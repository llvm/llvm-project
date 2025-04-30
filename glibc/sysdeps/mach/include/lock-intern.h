#ifndef _LOCK_INTERN_H
#include <mach/lock-intern.h>
#ifndef _ISOMAC
libc_hidden_proto (__spin_lock_locked)
libc_hidden_proto (__spin_lock)
libc_hidden_proto (__spin_lock_solid)
libc_hidden_proto (__spin_unlock)
libc_hidden_proto (__spin_try_lock)
libc_hidden_proto (__mutex_init)
libc_hidden_proto (__mutex_lock)
libc_hidden_proto (__mutex_unlock)
libc_hidden_proto (__mutex_trylock)
#endif
#endif
