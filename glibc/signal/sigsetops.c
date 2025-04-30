/* Compatibility symbols for old versions of signal.h.
   Copyright (C) 2017-2021 Free Software Foundation, Inc.
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

#include <signal.h>
#include <sigsetops.h>
#include <shlib-compat.h>

/* These were formerly defined by <signal.h> as inline functions,
   so they require out-of-line compatibility definitions.  */
#if SHLIB_COMPAT (libc, GLIBC_2_0, GLIBC_2_26)

int
attribute_compat_text_section
__sigismember_compat (const __sigset_t *set, int sig)
{
  return __sigismember (set, sig);
}
compat_symbol (libc, __sigismember_compat, __sigismember, GLIBC_2_0);

int
attribute_compat_text_section
__sigaddset_compat (__sigset_t *set, int sig)
{
  __sigaddset (set, sig);
  return 0;
}
compat_symbol (libc, __sigaddset_compat, __sigaddset, GLIBC_2_0);

int
attribute_compat_text_section
__sigdelset_compat (__sigset_t *set, int sig)
{
  __sigdelset (set, sig);
  return 0;
}
compat_symbol (libc, __sigdelset_compat, __sigdelset, GLIBC_2_0);

#endif
