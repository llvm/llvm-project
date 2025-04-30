/* Linux setrlimit implementation (32 bits off_t).
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
   License along with the GNU C Library.  If not, see
   <https://www.gnu.org/licenses/>.  */

#include <sys/resource.h>
#include <sysdep.h>
#include <shlib-compat.h>

#if !__RLIM_T_MATCHES_RLIM64_T

/* The compatibility symbol is meant to match the old __NR_getrlimit syscall
   (with broken RLIM_INFINITY definition).  It should be provided iff
   __NR_getrlimit and __NR_ugetrlimit are both defined.  */
# ifndef __NR_ugetrlimit
#  undef SHLIB_COMPAT
#  define SHLIB_COMPAT(a, b, c) 0
# endif

int
__setrlimit (enum __rlimit_resource resource, const struct rlimit *rlim)
{
  struct rlimit64 rlim64;

  if (rlim->rlim_cur == RLIM_INFINITY)
    rlim64.rlim_cur = RLIM64_INFINITY;
  else
    rlim64.rlim_cur = rlim->rlim_cur;
  if (rlim->rlim_max == RLIM_INFINITY)
    rlim64.rlim_max = RLIM64_INFINITY;
  else
    rlim64.rlim_max = rlim->rlim_max;

  return INLINE_SYSCALL_CALL (prlimit64, 0, resource, &rlim64, NULL);
}

libc_hidden_def (__setrlimit)
# if SHLIB_COMPAT (libc, GLIBC_2_0, GLIBC_2_2)
strong_alias (__setrlimit, __setrlimit_1)
compat_symbol (libc, __setrlimit, setrlimit, GLIBC_2_0);
versioned_symbol (libc, __setrlimit_1, setrlimit, GLIBC_2_2);
# else
weak_alias (__setrlimit, setrlimit)
# endif

#endif /* __RLIM_T_MATCHES_RLIM64_T  */
