/* Symbol version management.  ia64 version.
   Copyright (C) 2021 Free Software Foundation, Inc.
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

#ifndef _LIBC_SYMVER_H

#include <sysdeps/generic/libc-symver.h>

/* ia64 recognizes loc1 as a register name.  Add the # suffix to all
   symbol references.  */
#if !defined (__ASSEMBLER__) && SYMVER_NEEDS_ALIAS
#undef _set_symbol_version_2
# define _set_symbol_version_2(real, alias, name_version) \
  __asm__ (".globl " #alias "#\n\t"                         \
           ".equiv " #alias ", " #real "#\n\t"              \
           ".symver " #alias "#," name_version)
#endif

#endif /* _LIBC_SYMVER_H */
