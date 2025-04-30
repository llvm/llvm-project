/* non altivec (old) version of setjmp for PowerPC.
   Copyright (C) 1995-2021 Free Software Foundation, Inc.
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

#include <shlib-compat.h>
#include <libc-symbols.h>

#if !IS_IN (libc)
/* Build a non-versioned object for rtld-*.  */
# define __sigsetjmp_symbol __sigsetjmp
# define __sigjmp_save_symbol __sigjmp_save
# include "setjmp-common.S"

#else /* IS_IN (libc) */
/* Build a versioned object for libc.  */
versioned_symbol (libc, __vmx__sigsetjmp, __sigsetjmp, GLIBC_2_3_4)
# define __sigsetjmp_symbol __vmx__sigsetjmp
# define __sigjmp_save_symbol __vmx__sigjmp_save
# include "setjmp-common.S"
libc_hidden_ver (__vmx__sigsetjmp, __sigsetjmp)

# if defined SHARED && SHLIB_COMPAT (libc, GLIBC_2_0, GLIBC_2_3_4)
#  define __NO_VMX__
#  undef __sigsetjmp_symbol
#  undef __sigjmp_save_symbol
#  undef JB_SIZE
compat_symbol (libc, __novmx__sigsetjmp, __sigsetjmp, GLIBC_2_0)
#  define __sigsetjmp_symbol __novmx__sigsetjmp
#  define __sigjmp_save_symbol __novmx__sigjmp_save
#  include "setjmp-common.S"
# endif
#endif /* IS_IN (libc) */
