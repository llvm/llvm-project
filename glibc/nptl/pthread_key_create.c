 /* Copyright (C) 2002-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Ulrich Drepper <drepper@redhat.com>, 2002.

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

#include <errno.h>
#include "pthreadP.h"
#include <atomic.h>
#include <shlib-compat.h>

int
___pthread_key_create (pthread_key_t *key, void (*destr) (void *))
{
  /* Find a slot in __pthread_keys which is unused.  */
  for (size_t cnt = 0; cnt < PTHREAD_KEYS_MAX; ++cnt)
    {
      uintptr_t seq = __pthread_keys[cnt].seq;

      if (KEY_UNUSED (seq) && KEY_USABLE (seq)
	  /* We found an unused slot.  Try to allocate it.  */
	  && ! atomic_compare_and_exchange_bool_acq (&__pthread_keys[cnt].seq,
						     seq + 1, seq))
	{
	  /* Remember the destructor.  */
	  __pthread_keys[cnt].destr = destr;

	  /* Return the key to the caller.  */
	  *key = cnt;

	  /* The call succeeded.  */
	  return 0;
	}
    }

  return EAGAIN;
}
versioned_symbol (libc, ___pthread_key_create, __pthread_key_create,
		  GLIBC_2_34);
libc_hidden_ver (___pthread_key_create, __pthread_key_create)

versioned_symbol (libc, ___pthread_key_create, pthread_key_create,
		  GLIBC_2_34);
#if OTHER_SHLIB_COMPAT (libpthread, GLIBC_2_0, GLIBC_2_34)
compat_symbol (libpthread, ___pthread_key_create, __pthread_key_create,
	       GLIBC_2_0);
compat_symbol (libpthread, ___pthread_key_create, pthread_key_create,
	       GLIBC_2_0);
#endif
