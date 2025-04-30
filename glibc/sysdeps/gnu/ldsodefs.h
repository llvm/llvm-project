/* Run-time dynamic linker data structures for loaded ELF shared objects.  GNU.
   Copyright (C) 2001-2021 Free Software Foundation, Inc.
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

#ifndef _GNU_LDSODEFS_H
#define _GNU_LDSODEFS_H 1

#include <elf.h>
#include <libc-abis.h>
#include <string.h>

/* Accept binaries which identify the binary as using GNU extensions.  */
#define VALID_ELF_HEADER(hdr,exp,size)	(memcmp (hdr, exp, size) == 0	\
					 || memcmp (hdr, expected2, size) == 0)
#define VALID_ELF_OSABI(osabi)		(osabi == ELFOSABI_SYSV \
					 || osabi == ELFOSABI_GNU)
#define VALID_ELF_ABIVERSION(osabi,ver) \
  (ver == 0 || (osabi == ELFOSABI_GNU && ver < LIBC_ABI_MAX))
#define MORE_ELF_HEADER_DATA \
  static const unsigned char expected2[EI_PAD] =	\
  {							\
    [EI_MAG0] = ELFMAG0,				\
    [EI_MAG1] = ELFMAG1,				\
    [EI_MAG2] = ELFMAG2,				\
    [EI_MAG3] = ELFMAG3,				\
    [EI_CLASS] = ELFW(CLASS),				\
    [EI_DATA] = byteorder,				\
    [EI_VERSION] = EV_CURRENT,				\
    [EI_OSABI] = ELFOSABI_GNU				\
  }

#include_next <ldsodefs.h>

#endif /* _GNU_LDSODEFS_H */
