/* AltiVec (new) version of setjmp for PowerPC.
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

#include <libc-symbols.h>
#include <rtld-global-offsets.h>
#include <shlib-compat.h>

#if !IS_IN (libc)
/* Build a non-versioned object for rtld-*.  */
# define setjmp_symbol setjmp
# define _setjmp_symbol _setjmp
# define __sigsetjmp_symbol __sigsetjmp
# define __sigjmp_save_symbol __sigjmp_save
# include "setjmp-common.S"

#else /* IS_IN (libc) */
/* Build a versioned object for libc.  */
versioned_symbol (libc, __vmxsetjmp, setjmp, GLIBC_2_3_4)
versioned_symbol (libc, __vmx_setjmp, _setjmp, GLIBC_2_3_4)
versioned_symbol (libc, __vmx__sigsetjmp, __sigsetjmp, GLIBC_2_3_4)
# define setjmp_symbol __vmxsetjmp
# define _setjmp_symbol __vmx_setjmp
# define __sigsetjmp_symbol __vmx__sigsetjmp
# define __sigjmp_save_symbol __vmx__sigjmp_save
# include "setjmp-common.S"
strong_alias (__vmxsetjmp, __vmx__setjmp)
strong_alias (__vmx__sigsetjmp, __setjmp)

# if defined SHARED && SHLIB_COMPAT (libc, GLIBC_2_3, GLIBC_2_3_4)
#  undef setjmp_symbol
#  undef _setjmp_symbol
#  undef __sigsetjmp_symbol
#  undef __sigjmp_save_symbol
#  undef JB_SIZE
#  define __NO_VMX__
compat_symbol (libc, __novmxsetjmp, setjmp, GLIBC_2_3)
compat_symbol (libc, __novmx_setjmp,_setjmp, GLIBC_2_3);
compat_symbol (libc, __novmx__sigsetjmp,__sigsetjmp, GLIBC_2_3)
#  define setjmp_symbol __novmxsetjmp
#  define _setjmp_symbol __novmx_setjmp
#  define __sigsetjmp_symbol __novmx__sigsetjmp
#  define __sigjmp_save_symbol __novmx__sigjmp_save
#  include "setjmp-common.S"
strong_alias (__novmxsetjmp, __novmx__setjmp)
# endif
#endif /* IS_IN (libc) */
