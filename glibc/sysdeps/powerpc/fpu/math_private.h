/* Private inline math functions for powerpc.
   Copyright (C) 2006-2021 Free Software Foundation, Inc.
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

#ifndef _PPC_MATH_PRIVATE_H_
#define _PPC_MATH_PRIVATE_H_

#include <sysdep.h>
#include <ldsodefs.h>
#include <dl-procinfo.h>

#include_next <math_private.h>

#ifdef _ARCH_PWR9

#if __GNUC_PREREQ (8, 0)
# define _GL_HAS_BUILTIN_ILOGB 1
#elif defined __has_builtin
# define _GL_HAS_BUILTIN_ILOGB __has_builtin (__builtin_vsx_scalar_extract_exp)
#else
# define _GL_HAS_BUILTIN_ILOGB 0
#endif

#define __builtin_test_dc_ilogbf __builtin_test_dc_ilogb
#define __builtin_ilogbf __builtin_ilogb

#define __builtin_test_dc_ilogb(x, y) \
        __builtin_vsx_scalar_test_data_class_dp(x, y)
#define __builtin_ilogb(x) __builtin_vsx_scalar_extract_exp(x) - 0x3ff

#define __builtin_test_dc_ilogbf128(x, y) \
        __builtin_vsx_scalar_test_data_class_qp(x, y)
#define __builtin_ilogbf128(x) __builtin_vsx_scalar_extract_expq(x) - 0x3fff

#if __HAVE_DISTINCT_FLOAT128
extern __always_inline _Float128
__ieee754_sqrtf128 (_Float128 __x)
{
  _Float128 __z;
  asm ("xssqrtqp %0,%1" : "=v" (__z) : "v" (__x));
  return __z;
}
#endif
#else /* !_ARCH_PWR9 */
#define _GL_HAS_BUILTIN_ILOGB 0
#endif

#endif /* _PPC_MATH_PRIVATE_H_ */
