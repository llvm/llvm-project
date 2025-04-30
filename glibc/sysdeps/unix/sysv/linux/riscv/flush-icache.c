/* RISC-V instruction cache flushing VDSO calls
   Copyright (C) 2017-2021 Free Software Foundation, Inc.

   This file is part of the GNU C Library.

   The GNU C Library is free software; you can redistribute it and/or
   modify it under the terms of the GNU Lesser General Public License as
   published by the Free Software Foundation; either version 2.1 of the
   License, or (at your option) any later version.

   The GNU C Library is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
   Lesser General Public License for more details.

   You should have received a copy of the GNU Lesser General Public
   License along with the GNU C Library; if not, see
   <https://www.gnu.org/licenses/>.  */

#include <dl-vdso.h>
#include <stdlib.h>
#include <atomic.h>
#include <sys/cachectl.h>
#if __has_include (<asm/syscalls.h>)
# include <asm/syscalls.h>
#else
# include <asm/unistd.h>
#endif
#include <sys/syscall.h>

typedef int (*func_type) (void *, void *, unsigned long int);

static int
__riscv_flush_icache_syscall (void *start, void *end, unsigned long int flags)
{
  return INLINE_SYSCALL (riscv_flush_icache, 3, start, end, flags);
}

static func_type
__lookup_riscv_flush_icache (void)
{
  func_type func = dl_vdso_vsym ("__vdso_flush_icache");

  /* If there is no vDSO entry then call the system call directly.  All Linux
     versions provide the vDSO entry, but QEMU's user-mode emulation doesn't
     provide a vDSO.  */
  if (!func)
    func = &__riscv_flush_icache_syscall;

  return func;
}

#ifdef SHARED

# define INIT_ARCH()
libc_ifunc (__riscv_flush_icache, __lookup_riscv_flush_icache ())

#else

int
__riscv_flush_icache (void *start, void *end, unsigned long int flags)
{
  static volatile func_type cached_func;

  func_type func = atomic_load_relaxed (&cached_func);

  if (!func)
    {
      func = __lookup_riscv_flush_icache ();
      atomic_store_relaxed (&cached_func, func);
    }

  return func (start, end, flags);
}

#endif
