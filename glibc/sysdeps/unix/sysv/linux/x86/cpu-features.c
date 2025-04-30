/* Initialize CPU feature data for Linux/x86.
   This file is part of the GNU C Library.
   Copyright (C) 2018-2021 Free Software Foundation, Inc.

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

#if CET_ENABLED
# include <sys/prctl.h>
# include <asm/prctl.h>

static inline int __attribute__ ((always_inline))
get_cet_status (void)
{
  unsigned long long cet_status[3];
  if (INTERNAL_SYSCALL_CALL (arch_prctl, ARCH_CET_STATUS, cet_status) == 0)
    return cet_status[0];
  return 0;
}

# ifndef SHARED
static inline void
x86_setup_tls (void)
{
  __libc_setup_tls ();
  THREAD_SETMEM (THREAD_SELF, header.feature_1, GL(dl_x86_feature_1));
}

#  define ARCH_SETUP_TLS() x86_setup_tls ()
# endif
#endif

#include <sysdeps/x86/cpu-features.c>
