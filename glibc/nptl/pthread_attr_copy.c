/* Deep copy of a pthread_attr_t object.
   Copyright (C) 2020-2021 Free Software Foundation, Inc.
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

#include <errno.h>
#include <pthreadP.h>
#include <stdlib.h>

int
__pthread_attr_copy (pthread_attr_t *target, const pthread_attr_t *source)
{
  /* Avoid overwriting *TARGET until all allocations have
     succeeded.  */
  union pthread_attr_transparent temp;
  temp.external = *source;

  /* Force new allocation.  This function has full ownership of temp.  */
  temp.internal.extension = NULL;

  int ret = 0;

  struct pthread_attr *isource = (struct pthread_attr *) source;

  if (isource->extension != NULL)
    {
      /* Propagate affinity mask information.  */
      if (isource->extension->cpusetsize > 0)
        ret = __pthread_attr_setaffinity_np (&temp.external,
                                             isource->extension->cpusetsize,
                                             isource->extension->cpuset);

      /* Propagate the signal mask information.  */
      if (ret == 0 && isource->extension->sigmask_set)
        ret = __pthread_attr_setsigmask_internal ((pthread_attr_t *) &temp,
                                                  &isource->extension->sigmask);
    }

  if (ret != 0)
    {
      /* Deallocate because we have ownership.  */
      __pthread_attr_destroy (&temp.external);
      return ret;
    }

  /* Transfer ownership.  *target is not assumed to have been
     initialized.  */
  *target = temp.external;
  return 0;
}
libc_hidden_def (__pthread_attr_copy)
