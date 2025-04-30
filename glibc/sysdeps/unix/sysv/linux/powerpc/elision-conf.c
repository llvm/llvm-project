/* elision-conf.c: Lock elision tunable parameters.
   Copyright (C) 2015-2021 Free Software Foundation, Inc.
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

#include "config.h"
#include <pthreadP.h>
#include <elision-conf.h>
#include <unistd.h>
#include <dl-procinfo.h>

#if HAVE_TUNABLES
# define TUNABLE_NAMESPACE elision
#endif
#include <elf/dl-tunables.h>

/* Reasonable initial tuning values, may be revised in the future.
   This is a conservative initial value.  */

struct elision_config __elision_aconf =
  {
    /* How many times to use a non-transactional lock after a transactional
       failure has occurred because the lock is already acquired.  Expressed
       in number of lock acquisition attempts.  */
    .skip_lock_busy = 3,
    /* How often to not attempt to use elision if a transaction aborted due
       to reasons other than other threads' memory accesses.  Expressed in
       number of lock acquisition attempts.  */
    .skip_lock_internal_abort = 3,
    /* How often to not attempt to use elision if a lock used up all retries
       without success.  Expressed in number of lock acquisition attempts.  */
    .skip_lock_out_of_tbegin_retries = 3,
    /* How often we retry using elision if there is chance for the transaction
       to finish execution (e.g., it wasn't aborted due to the lock being
       already acquired.  */
    .try_tbegin = 3,
    /* Same as SKIP_LOCK_INTERNAL_ABORT but for trylock.  */
    .skip_trylock_internal_abort = 3,
  };

#if HAVE_TUNABLES
static inline void
__always_inline
do_set_elision_enable (int32_t elision_enable)
{
  /* Enable elision if it's avaliable in hardware. It's not necessary to check
     if __libc_enable_secure isn't enabled since elision_enable will be set
     according to the default, which is disabled.  */
  if (elision_enable == 1)
    __pthread_force_elision = (GLRO (dl_hwcap2)
			       & PPC_FEATURE2_HAS_HTM) ? 1 : 0;
}

/* The pthread->elision_enable tunable is 0 or 1 indicating that elision
   should be disabled or enabled respectively.  The feature will only be used
   if it's supported by the hardware.  */

void
TUNABLE_CALLBACK (set_elision_enable) (tunable_val_t *valp)
{
  int32_t elision_enable = (int32_t) valp->numval;
  do_set_elision_enable (elision_enable);
}

#define TUNABLE_CALLBACK_FNDECL(__name, __type)			\
static inline void						\
__always_inline							\
do_set_elision_ ## __name (__type value)			\
{								\
  __elision_aconf.__name = value;				\
}								\
void								\
TUNABLE_CALLBACK (set_elision_ ## __name) (tunable_val_t *valp) \
{								\
  __type value = (__type) (valp)->numval;			\
  do_set_elision_ ## __name (value);				\
}

TUNABLE_CALLBACK_FNDECL (skip_lock_busy, int32_t);
TUNABLE_CALLBACK_FNDECL (skip_lock_internal_abort, int32_t);
TUNABLE_CALLBACK_FNDECL (skip_lock_out_of_tbegin_retries, int32_t);
TUNABLE_CALLBACK_FNDECL (try_tbegin, int32_t);
TUNABLE_CALLBACK_FNDECL (skip_trylock_internal_abort, int32_t);
#endif

/* Initialize elision.  */

void
__lll_elision_init (void)
{
#if HAVE_TUNABLES
  /* Elision depends on tunables and must be explicitly turned on by setting
     the appropriate tunable on a supported platform.  */

  TUNABLE_GET (enable, int32_t,
	       TUNABLE_CALLBACK (set_elision_enable));
  TUNABLE_GET (skip_lock_busy, int32_t,
	       TUNABLE_CALLBACK (set_elision_skip_lock_busy));
  TUNABLE_GET (skip_lock_internal_abort, int32_t,
	       TUNABLE_CALLBACK (set_elision_skip_lock_internal_abort));
  TUNABLE_GET (skip_lock_after_retries, int32_t,
	       TUNABLE_CALLBACK (set_elision_skip_lock_out_of_tbegin_retries));
  TUNABLE_GET (tries, int32_t,
	       TUNABLE_CALLBACK (set_elision_try_tbegin));
  TUNABLE_GET (skip_trylock_internal_abort, int32_t,
	       TUNABLE_CALLBACK (set_elision_skip_trylock_internal_abort));
#endif

  /* Linux from 3.9 through 4.2 do not abort HTM transaction on syscalls,
     instead it suspends the transaction and resumes it when returning to
     usercode.  The side-effects of the syscall will always remain visible,
     even if the transaction is aborted.  This is an issue when a transaction
     is used along with futex syscall, on pthread_cond_wait for instance,
     where futex might succeed but the transaction is rolled back leading
     the condition variable object in an inconsistent state.

     Glibc used to prevent it by always aborting a transaction before issuing
     a syscall.  Linux 4.2 also decided to abort active transaction in
     syscalls which makes the glibc workaround superflours.  Worse, glibc
     transaction abortions leads to a performance issues on recent kernels.

     So Lock Elision is just enabled when it has been explict set (either
     by tunables of by a configure switch) and if kernel aborts HTM
     transactions on syscalls (PPC_FEATURE2_HTM_NOSC)  */

  __pthread_force_elision = (__pthread_force_elision
			     && GLRO (dl_hwcap2) & PPC_FEATURE2_HTM_NOSC);

  if (!__pthread_force_elision)
    __elision_aconf.try_tbegin = 0; /* Disable elision on rwlocks.  */
}
