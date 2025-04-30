/* Copyright (C) 1994-2021 Free Software Foundation, Inc.
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

#ifndef _LOCK_INTERN_H
#define	_LOCK_INTERN_H

#include <sys/cdefs.h>
#if defined __USE_EXTERN_INLINES && defined _LIBC
# include <lowlevellock.h>
#endif

#ifndef _EXTERN_INLINE
#define _EXTERN_INLINE __extern_inline
#endif

/* The type of a spin lock variable.  */
typedef unsigned int __spin_lock_t;

/* Static initializer for spinlocks.  */
#define __SPIN_LOCK_INITIALIZER   LLL_LOCK_INITIALIZER

/* Initialize LOCK.  */

extern void __spin_lock_init (__spin_lock_t *__lock);

#if defined __USE_EXTERN_INLINES && defined _LIBC
_EXTERN_INLINE void
__spin_lock_init (__spin_lock_t *__lock)
{
  *__lock = __SPIN_LOCK_INITIALIZER;
}
#endif


/* Lock LOCK, blocking if we can't get it.  */
extern void __spin_lock_solid (__spin_lock_t *__lock);

/* Lock the spin lock LOCK.  */

extern void __spin_lock (__spin_lock_t *__lock);

#if defined __USE_EXTERN_INLINES && defined _LIBC
_EXTERN_INLINE void
__spin_lock (__spin_lock_t *__lock)
{
  __lll_lock (__lock, 0);
}
#endif

/* Unlock LOCK.  */
extern void __spin_unlock (__spin_lock_t *__lock);

#if defined __USE_EXTERN_INLINES && defined _LIBC
_EXTERN_INLINE void
__spin_unlock (__spin_lock_t *__lock)
{
  __lll_unlock (__lock, 0);
}
#endif

/* Try to lock LOCK; return nonzero if we locked it, zero if another has.  */
extern int __spin_try_lock (__spin_lock_t *__lock);

#if defined __USE_EXTERN_INLINES && defined _LIBC
_EXTERN_INLINE int
__spin_try_lock (__spin_lock_t *__lock)
{
  return (__lll_trylock (__lock) == 0);
}
#endif

/* Return nonzero if LOCK is locked.  */
extern int __spin_lock_locked (__spin_lock_t *__lock);

#if defined __USE_EXTERN_INLINES && defined _LIBC
_EXTERN_INLINE int
__spin_lock_locked (__spin_lock_t *__lock)
{
  return (*(volatile __spin_lock_t *)__lock != 0);
}
#endif

/* Name space-clean internal interface to mutex locks.  */
struct mutex {
	__spin_lock_t __held;
	__spin_lock_t __lock;
	const char *__name;
	void *__head, *__tail;
	void *__holder;
};

#define MUTEX_INITIALIZER { __SPIN_LOCK_INITIALIZER }

/* Initialize the newly allocated mutex lock LOCK for further use.  */
extern void __mutex_init (void *__lock);

/* Lock the mutex lock LOCK.  */

extern void __mutex_lock (void *__lock);

#if defined __USE_EXTERN_INLINES && defined _LIBC
_EXTERN_INLINE void
__mutex_lock (void *__lock)
{
  __spin_lock ((__spin_lock_t *)__lock);
}
#endif

/* Unlock the mutex lock LOCK.  */

extern void __mutex_unlock (void *__lock);

#if defined __USE_EXTERN_INLINES && defined _LIBC
_EXTERN_INLINE void
__mutex_unlock (void *__lock)
{
  __spin_unlock ((__spin_lock_t *)__lock);
}
#endif


extern int __mutex_trylock (void *__lock);

#if defined __USE_EXTERN_INLINES && defined _LIBC
_EXTERN_INLINE int
__mutex_trylock (void *__lock)
{
  return (__spin_try_lock ((__spin_lock_t *)__lock));
}
#endif

#endif /* lock-intern.h */
