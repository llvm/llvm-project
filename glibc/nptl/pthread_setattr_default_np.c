/* Set the default attributes to be used by pthread_create in the process.
   Copyright (C) 2013-2021 Free Software Foundation, Inc.
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
#include <stdlib.h>
#include <pthreadP.h>
#include <string.h>
#include <shlib-compat.h>

int
__pthread_setattr_default_np (const pthread_attr_t *in)
{
  const struct pthread_attr *real_in;
  int ret;

  real_in = (struct pthread_attr *) in;

  /* Catch invalid values.  */
  int policy = real_in->schedpolicy;
  ret = check_sched_policy_attr (policy);
  if (ret)
    return ret;

  const struct sched_param *param = &real_in->schedparam;
  if (param->sched_priority > 0)
    {
      ret = check_sched_priority_attr (param->sched_priority, policy);
      if (ret)
	return ret;
    }

  /* stacksize == 0 is fine.  It means that we don't change the current
     value.  */
  if (real_in->stacksize != 0)
    {
      ret = check_stacksize_attr (real_in->stacksize);
      if (ret)
	return ret;
    }

  /* Having a default stack address is wrong.  */
  if (real_in->flags & ATTR_FLAG_STACKADDR)
    return EINVAL;

  union pthread_attr_transparent temp;
  ret = __pthread_attr_copy (&temp.external, in);
  if (ret != 0)
    return ret;

  /* Now take the lock because we start accessing
     __default_pthread_attr.  */
  lll_lock (__default_pthread_attr_lock, LLL_PRIVATE);

  /* Preserve the previous stack size (see above).  */
  if (temp.internal.stacksize == 0)
    temp.internal.stacksize = __default_pthread_attr.internal.stacksize;

  /* Destroy the old attribute structure because it will be
     overwritten.  */
  __pthread_attr_destroy (&__default_pthread_attr.external);

  /* __default_pthread_attr takes ownership, so do not free
     attrs.internal after this point.  */
  __default_pthread_attr = temp;

  lll_unlock (__default_pthread_attr_lock, LLL_PRIVATE);
  return ret;
}
versioned_symbol (libc, __pthread_setattr_default_np,
		  pthread_setattr_default_np, GLIBC_2_34);
#if OTHER_SHLIB_COMPAT (libpthread, GLIBC_2_18, GLIBC_2_34)
compat_symbol (libc, __pthread_setattr_default_np,
	       pthread_setattr_default_np, GLIBC_2_18);
#endif

/* This is placed in the same file as pthread_setattr_default_np
   because only this function can trigger allocation of attribute
   data.  This way, the function is automatically defined for all the
   cases when it is needed in static builds.  */
void
__default_pthread_attr_freeres (void)
{
  __pthread_attr_destroy (&__default_pthread_attr.external);
}
