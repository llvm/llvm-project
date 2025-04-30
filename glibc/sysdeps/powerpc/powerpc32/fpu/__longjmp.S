/* AltiVec/VMX (new) version of __longjmp for PowerPC.
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
# define __longjmp_symbol __longjmp
# include "__longjmp-common.S"

#else /* IS_IN (libc) */
strong_alias (__vmx__longjmp, __longjmp)
# define __longjmp_symbol  __vmx__longjmp
# include "__longjmp-common.S"

# if defined SHARED && SHLIB_COMPAT (libc, GLIBC_2_0, GLIBC_2_3_4)
#  define __NO_VMX__
#  undef JB_SIZE
#  undef __longjmp_symbol
#  define __longjmp_symbol  __novmx__longjmp
#  include "__longjmp-common.S"
# endif
#endif /* IS_IN (libc) */
