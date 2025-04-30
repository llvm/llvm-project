#ifndef	_PTHREAD_H
#include_next <pthread.h>
#ifndef _ISOMAC

# define ARCH_MIN_GUARD_SIZE 0

# if defined __USE_EXTERN_INLINES && defined _LIBC && !IS_IN (libsupport)
#  include <bits/spin-lock-inline.h>

__extern_inline int
pthread_spin_destroy (pthread_spinlock_t *__lock)
{
  return __pthread_spin_destroy (__lock);
}

__extern_inline int
pthread_spin_init (pthread_spinlock_t *__lock, int __pshared)
{
  return __pthread_spin_init (__lock, __pshared);
}

__extern_inline int
pthread_spin_lock (pthread_spinlock_t *__lock)
{
  return __pthread_spin_lock (__lock);
}

__extern_inline int
pthread_spin_trylock (pthread_spinlock_t *__lock)
{
  return __pthread_spin_trylock (__lock);
}

__extern_inline int
pthread_spin_unlock (pthread_spinlock_t *__lock)
{
  return __pthread_spin_unlock (__lock);
}
# endif
#endif
#endif
