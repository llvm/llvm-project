/* Compute argument of complex float value.
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

#define __cargf __cargf_not_defined
#define cargf cargf_not_defined

#include <complex.h>
#include <math.h>
#include "cfloat-compat.h"

#undef __cargf
#undef cargf

float
__c1_cargf (c1_cfloat_decl (x))
{
  return __atan2f (c1_cfloat_imag (x), c1_cfloat_real (x));
}

float
__c2_cargf (c2_cfloat_decl (x))
{
  return __atan2f (c2_cfloat_imag (x), c2_cfloat_real (x));
}

cfloat_versions (carg);
