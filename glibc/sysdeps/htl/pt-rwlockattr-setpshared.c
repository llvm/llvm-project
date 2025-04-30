/* pthread_rwlockattr_setpshared.  Generic version.
   Copyright (C) 2002-2021 Free Software Foundation, Inc.
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
   License along with the GNU C Library;  if not, see
   <https://www.gnu.org/licenses/>.  */

#include <pthread.h>
#include <pt-internal.h>

int
pthread_rwlockattr_setpshared (pthread_rwlockattr_t *attr, int pshared)
{
  switch (pshared)
    {
    case PTHREAD_PROCESS_PRIVATE:
      attr->__pshared = pshared;
      return 0;

    case PTHREAD_PROCESS_SHARED:
      return ENOTSUP;

    default:
      return EINVAL;
    }
}
stub_warning (pthread_rwlockattr_setpshared)
