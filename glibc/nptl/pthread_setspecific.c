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
#include <stdlib.h>
#include "pthreadP.h"
#include <shlib-compat.h>

int
___pthread_setspecific (pthread_key_t key, const void *value)
{
  struct pthread *self;
  unsigned int idx1st;
  unsigned int idx2nd;
  struct pthread_key_data *level2;
  unsigned int seq;

  self = THREAD_SELF;

  /* Special case access to the first 2nd-level block.  This is the
     usual case.  */
  if (__glibc_likely (key < PTHREAD_KEY_2NDLEVEL_SIZE))
    {
      /* Verify the key is sane.  */
      if (KEY_UNUSED ((seq = __pthread_keys[key].seq)))
	/* Not valid.  */
	return EINVAL;

      level2 = &self->specific_1stblock[key];

      /* Remember that we stored at least one set of data.  */
      if (value != NULL)
	THREAD_SETMEM (self, specific_used, true);
    }
  else
    {
      if (key >= PTHREAD_KEYS_MAX
	  || KEY_UNUSED ((seq = __pthread_keys[key].seq)))
	/* Not valid.  */
	return EINVAL;

      idx1st = key / PTHREAD_KEY_2NDLEVEL_SIZE;
      idx2nd = key % PTHREAD_KEY_2NDLEVEL_SIZE;

      /* This is the second level array.  Allocate it if necessary.  */
      level2 = THREAD_GETMEM_NC (self, specific, idx1st);
      if (level2 == NULL)
	{
	  if (value == NULL)
	    /* We don't have to do anything.  The value would in any case
	       be NULL.  We can save the memory allocation.  */
	    return 0;

	  level2
	    = (struct pthread_key_data *) calloc (PTHREAD_KEY_2NDLEVEL_SIZE,
						  sizeof (*level2));
	  if (level2 == NULL)
	    return ENOMEM;

	  THREAD_SETMEM_NC (self, specific, idx1st, level2);
	}

      /* Pointer to the right array element.  */
      level2 = &level2[idx2nd];

      /* Remember that we stored at least one set of data.  */
      THREAD_SETMEM (self, specific_used, true);
    }

  /* Store the data and the sequence number so that we can recognize
     stale data.  */
  level2->seq = seq;
  level2->data = (void *) value;

  return 0;
}
versioned_symbol (libc, ___pthread_setspecific, pthread_setspecific,
		  GLIBC_2_34);
libc_hidden_ver (___pthread_setspecific, __pthread_setspecific)
#ifndef SHARED
strong_alias (___pthread_setspecific, __pthread_setspecific)
#endif

#if OTHER_SHLIB_COMPAT (libc, GLIBC_2_0, GLIBC_2_34)
compat_symbol (libpthread, ___pthread_setspecific, __pthread_setspecific,
	       GLIBC_2_0);
compat_symbol (libpthread, ___pthread_setspecific, pthread_setspecific,
	       GLIBC_2_0);
#endif
