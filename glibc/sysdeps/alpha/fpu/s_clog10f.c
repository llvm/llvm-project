/* Return base 10 logarithm of complex float value.
   Copyright (C) 2004-2021 Free Software Foundation, Inc.
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

#define __clog10f __clog10f_not_defined
#define clog10f clog10f_not_defined

#include <complex.h>
#include <math.h>
#include <libm-alias-float.h>

#undef __clog10f
#undef clog10f

static _Complex float internal_clog10f (_Complex float x);

#define M_DECL_FUNC(f) internal_clog10f
#include <math-type-macros-float.h>

/* Disable any aliasing from base template.  */
#undef declare_mgen_alias
#define declare_mgen_alias(__to, __from)

#include <math/s_clog10_template.c>
#include "cfloat-compat.h"

c1_cfloat_rettype
__c1_clog10f (c1_cfloat_decl (x))
{
  _Complex float r = internal_clog10f (c1_cfloat_value (x));
  return c1_cfloat_return (r);
}

c2_cfloat_rettype
__c2_clog10f (c2_cfloat_decl (x))
{
  _Complex float r = internal_clog10f (c2_cfloat_value (x));
  return c2_cfloat_return (r);
}

/* Ug.  __clog10f was exported from GLIBC_2.1.  This is the only
   complex function whose double-underscore symbol was exported,
   so we get to handle that specially.  */
#if SHLIB_COMPAT (libm, GLIBC_2_1, GLIBC_2_3_4)
strong_alias (__c1_clog10f, __c1_clog10f_2);
compat_symbol (libm, __c1_clog10f, clog10f, GLIBC_2_1);
compat_symbol (libm, __c1_clog10f_2, __clog10f, GLIBC_2_1);
#endif
versioned_symbol (libm, __c2_clog10f, clog10f, GLIBC_2_3_4);
extern typeof(__c2_clog10f) __clog10f attribute_hidden;
strong_alias (__c2_clog10f, __clog10f)
libm_alias_float_other (__c2_clog10, clog10)
