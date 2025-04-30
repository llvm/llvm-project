/* Run-time dynamic linker data structures for loaded ELF shared objects.
   Copyright (C) 2010-2021 Free Software Foundation, Inc.
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

#ifndef _ARM_LINUX_LDSODEFS_H
#define _ARM_LINUX_LDSODEFS_H 1

#include_next <ldsodefs.h>

#undef VALID_ELF_HEADER
#undef VALID_ELF_OSABI
#undef MORE_ELF_HEADER_DATA

#define EXTRA_OSABI ELFOSABI_ARM_AEABI

#ifdef __ARM_PCS_VFP
# define VALID_FLOAT_ABI(x) \
  ((EF_ARM_EABI_VERSION ((x)) != EF_ARM_EABI_VER5)	\
   || !((x) & EF_ARM_ABI_FLOAT_SOFT))
#else
# define VALID_FLOAT_ABI(x) \
  ((EF_ARM_EABI_VERSION ((x)) != EF_ARM_EABI_VER5)	\
   || !((x) & EF_ARM_ABI_FLOAT_HARD))
#endif

#undef VALID_ELF_HEADER
#define VALID_ELF_HEADER(hdr,exp,size)		\
  ((memcmp (hdr, exp, size) == 0		\
    || memcmp (hdr, expected2, size) == 0	\
    || memcmp (hdr, expected3, size) == 0)	\
   && VALID_FLOAT_ABI (ehdr->e_flags))
#define VALID_ELF_OSABI(osabi)		(osabi == ELFOSABI_SYSV		\
					 || osabi == ELFOSABI_GNU	\
					 || osabi == EXTRA_OSABI)
#define MORE_ELF_HEADER_DATA				\
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
  };							\
  static const unsigned char expected3[EI_PAD] =	\
  {							\
    [EI_MAG0] = ELFMAG0,				\
    [EI_MAG1] = ELFMAG1,				\
    [EI_MAG2] = ELFMAG2,				\
    [EI_MAG3] = ELFMAG3,				\
    [EI_CLASS] = ELFW(CLASS),				\
    [EI_DATA] = byteorder,				\
    [EI_VERSION] = EV_CURRENT,				\
    [EI_OSABI] = EXTRA_OSABI				\
  }

#endif
