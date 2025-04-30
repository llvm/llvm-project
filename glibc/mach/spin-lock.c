#define __USE_EXTERN_INLINES 1
#define _EXTERN_INLINE /* Empty to define the real functions.  */
#include "spin-lock.h"

weak_alias (__spin_lock_init, spin_lock_init);
libc_hidden_def (__spin_lock_locked);
weak_alias (__spin_lock_locked, spin_lock_locked);
libc_hidden_def (__spin_lock);
weak_alias (__spin_lock, spin_lock);
libc_hidden_def (__spin_unlock);
weak_alias (__spin_unlock, spin_unlock);
libc_hidden_def (__spin_try_lock);
weak_alias (__spin_try_lock, spin_try_lock);
libc_hidden_def (__mutex_lock);
libc_hidden_def (__mutex_unlock);
libc_hidden_def (__mutex_trylock);
