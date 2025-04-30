/* Copyright (C) 2001-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Ulrich Drepper <drepper@redhat.com>, 2001.

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

#include <netdb.h>
#include <pthread.h>
#include <shlib-compat.h>
#include <gai_misc.h>


int
__gai_cancel (struct gaicb *gaicbp)
{
  int result = 0;
  int status;

  /* Request the mutex.  */
  __pthread_mutex_lock (&__gai_requests_mutex);

  /* Find the request among those queued but not yet running.  */
  status = __gai_remove_request (gaicbp);
  if (status == 0)
    result = EAI_CANCELED;
  else if (status > 0)
    result = EAI_NOTCANCELED;
  else
    result = EAI_ALLDONE;

  /* Release the mutex.  */
  __pthread_mutex_unlock (&__gai_requests_mutex);

  return result;
}
#if PTHREAD_IN_LIBC
versioned_symbol (libc, __gai_cancel, gai_cancel, GLIBC_2_34);

# if OTHER_SHLIB_COMPAT (libanl, GLIBC_2_2_3, GLIBC_2_34)
compat_symbol (libanl, __gai_cancel, gai_cancel, GLIBC_2_2_3);
# endif
#else /* !PTHREAD_IN_LIBC */
strong_alias (__gai_cancel, gai_cancel)
#endif /* !PTHREAD_IN_LIBC */
