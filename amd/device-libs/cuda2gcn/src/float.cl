/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#define ATTR __attribute__((const))

//-------- T __nv_finitef
ATTR int __nv_finitef(float x) { return isfinite(x); }

//-------- T __nv_isfinited
ATTR int __nv_isfinited(double x) { return isfinite(x); }

//-------- T __nv_isinfd
ATTR int __nv_isinfd(double x) { return isinf(x); }

//-------- T __nv_isinff
ATTR int __nv_isinff(float x) { return isinf(x); }

//-------- T __nv_isnand
ATTR int __nv_isnand(double x) { return isnan(x); }

//-------- T __nv_isnanf
ATTR int __nv_isnanf(float x) { return isnan(x); }

//-------- T __nv_nan
ATTR double __nv_nan(char *tagp) { return __builtin_nan(tagp); }

//-------- T __nv_nanf
ATTR float __nv_nanf(char *tagp) { return __builtin_nan(tagp); }

