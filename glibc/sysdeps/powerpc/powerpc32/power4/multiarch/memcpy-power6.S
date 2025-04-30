/* Optimized memcpy implementation for PowerPC32 on POWER6.
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

#undef EALIGN
#define EALIGN(name, alignt, words)				\
 .globl C_SYMBOL_NAME(__memcpy_power6);				\
 .type C_SYMBOL_NAME(__memcpy_power6),@function;		\
 .align ALIGNARG(alignt);					\
 EALIGN_W_##words;						\
 C_LABEL(__memcpy_power6)					\
 cfi_startproc;

#undef END
#define END(name)						\
 cfi_endproc;							\
 ASM_SIZE_DIRECTIVE(__memcpy_power6)

#undef libc_hidden_builtin_def
#define libc_hidden_builtin_def(name)

#include <sysdeps/powerpc/powerpc32/power6/memcpy.S>
