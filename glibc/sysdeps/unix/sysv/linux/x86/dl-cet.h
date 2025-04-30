/* Linux/x86 CET initializers function.
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

#include <sys/prctl.h>
#include <asm/prctl.h>

static inline int __attribute__ ((always_inline))
dl_cet_disable_cet (unsigned int cet_feature)
{
  return (int) INTERNAL_SYSCALL_CALL (arch_prctl, ARCH_CET_DISABLE,
				      cet_feature);
}

static inline int __attribute__ ((always_inline))
dl_cet_lock_cet (void)
{
  return (int) INTERNAL_SYSCALL_CALL (arch_prctl, ARCH_CET_LOCK, 0);
}
