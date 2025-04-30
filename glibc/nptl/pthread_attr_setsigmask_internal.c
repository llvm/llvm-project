/* Set the signal mask in a POSIX thread attribute.  Internal variant.
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

#include <pthreadP.h>
#include <internal-signals.h>

int
__pthread_attr_setsigmask_internal (pthread_attr_t *attr,
                                    const sigset_t *sigmask)
{
  struct pthread_attr *iattr = (struct pthread_attr *) attr;

  if (sigmask == NULL)
    {
      /* Mark the signal mask as unset if it is present.  */
      if (iattr->extension != NULL)
        iattr->extension->sigmask_set = false;
      return 0;
    }

  int ret = __pthread_attr_extension (iattr);
  if (ret != 0)
    return ret;

  iattr->extension->sigmask = *sigmask;
  iattr->extension->sigmask_set = true;

  return 0;
}
libc_hidden_def (__pthread_attr_setsigmask_internal)
