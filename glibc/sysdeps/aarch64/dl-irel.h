/* Machine-dependent ELF indirect relocation inline functions.
   AArch64 version.
   Copyright (C) 2012-2021 Free Software Foundation, Inc.
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

#ifndef _DL_IREL_H
#define _DL_IREL_H

#include <stdio.h>
#include <unistd.h>
#include <ldsodefs.h>
#include <sysdep.h>
#include <sys/ifunc.h>

#define ELF_MACHINE_IRELA	1

static inline ElfW(Addr)
__attribute ((always_inline))
elf_ifunc_invoke (ElfW(Addr) addr)
{
  __ifunc_arg_t arg;

  arg._size = sizeof (arg);
  arg._hwcap = GLRO(dl_hwcap);
  arg._hwcap2 = GLRO(dl_hwcap2);
  return ((ElfW(Addr) (*) (uint64_t, const __ifunc_arg_t *)) (addr))
	 (GLRO(dl_hwcap) | _IFUNC_ARG_HWCAP, &arg);
}

static inline void
__attribute ((always_inline))
elf_irela (const ElfW(Rela) *reloc)
{
  ElfW(Addr) *const reloc_addr = (void *) reloc->r_offset;
  const unsigned long int r_type = ELFW(R_TYPE) (reloc->r_info);

  if (__glibc_likely (r_type == AARCH64_R(IRELATIVE)))
    {
      ElfW(Addr) value = elf_ifunc_invoke (reloc->r_addend);
      *reloc_addr = value;
    }
  else
    __libc_fatal ("Unexpected reloc type in static binary.\n");
}

#endif
