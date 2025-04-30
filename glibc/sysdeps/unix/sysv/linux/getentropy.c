/* Implementation of getentropy based on the getrandom system call.
   Copyright (C) 2016-2021 Free Software Foundation, Inc.
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

#include <sys/random.h>
#include <assert.h>
#include <errno.h>
#include <unistd.h>
#include <sysdep.h>

/* Write LENGTH bytes of randomness starting at BUFFER.  Return 0 on
   success and -1 on failure.  */
int
getentropy (void *buffer, size_t length)
{
  /* The interface is documented to return EIO for buffer lengths
     longer than 256 bytes.  */
  if (length > 256)
    {
      __set_errno (EIO);
      return -1;
    }

  /* Try to fill the buffer completely.  Even with the 256 byte limit
     above, we might still receive an EINTR error (when blocking
     during boot).  */
  void *end = buffer + length;
  while (buffer < end)
    {
      /* NB: No cancellation point.  */
      ssize_t bytes = INLINE_SYSCALL_CALL (getrandom, buffer, end - buffer, 0);
      if (bytes < 0)
        {
          if (errno == EINTR)
            /* Try again if interrupted by a signal.  */
            continue;
          else
            return -1;
        }
      if (bytes == 0)
        {
          /* No more bytes available.  This should not happen under
             normal circumstances.  */
          __set_errno (EIO);
          return -1;
        }
      /* Try again in case of a short read.  */
      buffer += bytes;
    }
  return 0;
}
