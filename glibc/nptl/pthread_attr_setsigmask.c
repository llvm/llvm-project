/* Set the signal mask in a POSIX thread attribute.  Public variant.
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
pthread_attr_setsigmask_np (pthread_attr_t *attr, const sigset_t *sigmask)
{
  int ret = __pthread_attr_setsigmask_internal (attr, sigmask);
  if (ret != 0)
    return ret;

  /* Filter out internal signals.  */
  struct pthread_attr *iattr = (struct pthread_attr *) attr;
  __clear_internal_signals (&iattr->extension->sigmask);

  return 0;
}
