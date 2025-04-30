/* Optimized strcasecmp implementation for PowerPC32.
   Copyright (C) 2013-2021 Free Software Foundation, Inc.
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

#include <sysdep.h>

#undef ENTRY
#define ENTRY(name)						\
 .globl C_SYMBOL_NAME(__strcasecmp_power7);			\
 .type C_SYMBOL_NAME(__strcasecmp_power7),@function;		\
 C_LABEL(__strcasecmp_power7)					\
 cfi_startproc;

#undef END
#define END(name)						\
 cfi_endproc;							\
 ASM_SIZE_DIRECTIVE(__strcasecmp_power7)

#undef weak_alias
#define weak_alias(name, alias)

#undef libc_hidden_builtin_def
#define libc_hidden_builtin_def(name)

#include <sysdeps/powerpc/powerpc32/power7/strcasecmp.S>
