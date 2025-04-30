/* ldconfig default paths and libraries.  Linux/C-SKY version.
   Copyright (C) 2018-2021 Free Software Foundation, Inc.
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

#include <sysdeps/generic/ldconfig.h>

#define LD_SO_PREFIX "/lib/ld-linux-"
#define LD_SO_SUFFIX ".so.1"

#ifdef __CSKY_HARD_FLOAT__
# define LD_SO_ABI "cskyv2-hf"
#else
# define LD_SO_ABI "cskyv2"
#endif

#define SYSDEP_KNOWN_INTERPRETER_NAMES	\
  { LD_SO_PREFIX LD_SO_ABI LD_SO_SUFFIX, FLAG_ELF_LIBC6 },
#define SYSDEP_KNOWN_LIBRARY_NAMES	\
  { "libc.so.6", FLAG_ELF_LIBC6 },	\
  { "libm.so.6", FLAG_ELF_LIBC6 },
