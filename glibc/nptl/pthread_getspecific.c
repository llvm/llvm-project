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

#include <stdlib.h>
#include "pthreadP.h"
#include <shlib-compat.h>

void *
___pthread_getspecific (pthread_key_t key)
{
  struct pthread_key_data *data;

  /* Special case access to the first 2nd-level block.  This is the
     usual case.  */
  if (__glibc_likely (key < PTHREAD_KEY_2NDLEVEL_SIZE))
    data = &THREAD_SELF->specific_1stblock[key];
  else
    {
      /* Verify the key is sane.  */
      if (key >= PTHREAD_KEYS_MAX)
	/* Not valid.  */
	return NULL;

      unsigned int idx1st = key / PTHREAD_KEY_2NDLEVEL_SIZE;
      unsigned int idx2nd = key % PTHREAD_KEY_2NDLEVEL_SIZE;

      /* If the sequence number doesn't match or the key cannot be defined
	 for this thread since the second level array is not allocated
	 return NULL, too.  */
      struct pthread_key_data *level2 = THREAD_GETMEM_NC (THREAD_SELF,
							  specific, idx1st);
      if (level2 == NULL)
	/* Not allocated, therefore no data.  */
	return NULL;

      /* There is data.  */
      data = &level2[idx2nd];
    }

  void *result = data->data;
  if (result != NULL)
    {
      uintptr_t seq = data->seq;

      if (__glibc_unlikely (seq != __pthread_keys[key].seq))
	result = data->data = NULL;
    }

  return result;
}
versioned_symbol (libc, ___pthread_getspecific, pthread_getspecific,
		  GLIBC_2_34);
libc_hidden_ver (___pthread_getspecific, __pthread_getspecific)
#ifndef SHARED
strong_alias (___pthread_getspecific, __pthread_getspecific)
#endif

#if OTHER_SHLIB_COMPAT (libc, GLIBC_2_0, GLIBC_2_34)
compat_symbol (libpthread, ___pthread_getspecific, __pthread_getspecific,
	       GLIBC_2_0);
compat_symbol (libpthread, ___pthread_getspecific, pthread_getspecific,
	       GLIBC_2_0);
#endif
