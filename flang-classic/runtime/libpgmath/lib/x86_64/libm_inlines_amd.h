/* ============================================================
Copyright (c) 2002-2015 Advanced Micro Devices, Inc.

All rights reserved.

Redistribution and  use in source and binary  forms, with or
without  modification,  are   permitted  provided  that  the
following conditions are met:

+ Redistributions  of source  code  must  retain  the  above
  copyright  notice,   this  list  of   conditions  and  the
  following disclaimer.

+ Redistributions  in binary  form must reproduce  the above
  copyright  notice,   this  list  of   conditions  and  the
  following  disclaimer in  the  documentation and/or  other
  materials provided with the distribution.

+ Neither the  name of Advanced Micro Devices,  Inc. nor the
  names  of  its contributors  may  be  used  to endorse  or
  promote  products  derived   from  this  software  without
  specific prior written permission.

THIS  SOFTWARE  IS PROVIDED  BY  THE  COPYRIGHT HOLDERS  AND
CONTRIBUTORS "AS IS" AND  ANY EXPRESS OR IMPLIED WARRANTIES,
INCLUDING,  BUT NOT  LIMITED TO,  THE IMPLIED  WARRANTIES OF
MERCHANTABILITY  AND FITNESS  FOR A  PARTICULAR  PURPOSE ARE
DISCLAIMED.  IN  NO  EVENT  SHALL  ADVANCED  MICRO  DEVICES,
INC.  OR CONTRIBUTORS  BE LIABLE  FOR ANY  DIRECT, INDIRECT,
INCIDENTAL,  SPECIAL,  EXEMPLARY,  OR CONSEQUENTIAL  DAMAGES
(INCLUDING,  BUT NOT LIMITED  TO, PROCUREMENT  OF SUBSTITUTE
GOODS  OR  SERVICES;  LOSS  OF  USE, DATA,  OR  PROFITS;  OR
BUSINESS INTERRUPTION)  HOWEVER CAUSED AND ON  ANY THEORY OF
LIABILITY,  WHETHER IN CONTRACT,  STRICT LIABILITY,  OR TORT
(INCLUDING NEGLIGENCE  OR OTHERWISE) ARISING IN  ANY WAY OUT
OF  THE  USE  OF  THIS  SOFTWARE, EVEN  IF  ADVISED  OF  THE
POSSIBILITY OF SUCH DAMAGE.

It is  licensee's responsibility  to comply with  any export
regulations applicable in licensee's jurisdiction.
============================================================ */
#ifndef LIBM_INLINES_AMD_H_INCLUDED
#define LIBM_INLINES_AMD_H_INCLUDED 1

#include "libm_util_amd.h"

/* Set defines for inline functions calling other inlines */
#if defined(USE_VAL_WITH_FLAGS) || defined(USE_VALF_WITH_FLAGS) ||             defined(USE_ZERO_WITH_FLAGS) || defined(USE_ZEROF_WITH_FLAGS) ||           defined(USE_NAN_WITH_FLAGS) || defined(USE_NANF_WITH_FLAGS) ||             defined(USE_INDEFINITE_WITH_FLAGS) ||                                      defined(USE_INDEFINITEF_WITH_FLAGS) || defined(USE_INFINITY_WITH_FLAGS) || defined(USE_INFINITYF_WITH_FLAGS) || defined(USE_SQRT_AMD_INLINE) ||       defined(USE_SQRTF_AMD_INLINE)

#undef USE_RAISE_FPSW_FLAGS
#define USE_RAISE_FPSW_FLAGS 1
#endif

#if defined(USE_SPLITDOUBLE)
/* Splits double x into exponent e and mantissa m, where 0.5 <= abs(m) < 1.0.
   Assumes that x is not zero, denormal, infinity or NaN, but these conditions
   are not checked */
static inline void
splitDouble(double x, int *e, double *m)
{
  __UINT8_T ux, uy;
  GET_BITS_DP64(x, ux);
  uy = ux;
  ux &= EXPBITS_DP64;
  ux >>= EXPSHIFTBITS_DP64;
  *e = (int)ux - EXPBIAS_DP64 + 1;
  uy = (uy & (SIGNBIT_DP64 | MANTBITS_DP64)) | HALFEXPBITS_DP64;
  PUT_BITS_DP64(uy, x);
  *m = x;
}
#endif /* USE_SPLITDOUBLE */

#if defined(USE_SPLITDOUBLE_2)
/* Splits double x into exponent e and mantissa m, where 1.0 <= abs(m) < 4.0.
   Assumes that x is not zero, denormal, infinity or NaN, but these conditions
   are not checked. Also assumes EXPBIAS_DP is odd. With this
   assumption, e will be even on exit. */
static inline void
splitDouble_2(double x, int *e, double *m)
{
  __UINT8_T ux, vx;
  GET_BITS_DP64(x, ux);
  vx = ux;
  ux &= EXPBITS_DP64;
  ux >>= EXPSHIFTBITS_DP64;
  if (ux & 1) {
    /* The exponent is odd */
    vx = (vx & (SIGNBIT_DP64 | MANTBITS_DP64)) | ONEEXPBITS_DP64;
    PUT_BITS_DP64(vx, x);
    *m = x;
    *e = ux - EXPBIAS_DP64;
  } else {
    /* The exponent is even */
    vx = (vx & (SIGNBIT_DP64 | MANTBITS_DP64)) | TWOEXPBITS_DP64;
    PUT_BITS_DP64(vx, x);
    *m = x;
    *e = ux - EXPBIAS_DP64 - 1;
  }
}
#endif /* USE_SPLITDOUBLE_2 */

#if defined(USE_SPLITFLOAT)
/* Splits float x into exponent e and mantissa m, where 0.5 <= abs(m) < 1.0.
   Assumes that x is not zero, denormal, infinity or NaN, but these conditions
   are not checked */
static inline void
splitFloat(float x, int *e, float *m)
{
  unsigned int ux, uy;
  GET_BITS_SP32(x, ux);
  uy = ux;
  ux &= EXPBITS_SP32;
  ux >>= EXPSHIFTBITS_SP32;
  *e = (int)ux - EXPBIAS_SP32 + 1;
  uy = (uy & (SIGNBIT_SP32 | MANTBITS_SP32)) | HALFEXPBITS_SP32;
  PUT_BITS_SP32(uy, x);
  *m = x;
}
#endif /* USE_SPLITFLOAT */

#if defined(USE_SCALEDOUBLE_1)
/* Scales the double x by 2.0**n.
   Assumes EMIN <= n <= EMAX, though this condition is not checked. */
static inline double
scaleDouble_1(double x, int n)
{
  double t;
  /* Construct the number t = 2.0**n */
  PUT_BITS_DP64(((__INT8_T)n + EXPBIAS_DP64) << EXPSHIFTBITS_DP64, t);
  return x * t;
}
#endif /* USE_SCALEDOUBLE_1 */

#if defined(USE_SCALEDOUBLE_2)
/* Scales the double x by 2.0**n.
   Assumes 2*EMIN <= n <= 2*EMAX, though this condition is not checked. */
static inline double
scaleDouble_2(double x, int n)
{
  double t1, t2;
  int n1, n2;
  n1 = n / 2;
  n2 = n - n1;
  /* Construct the numbers t1 = 2.0**n1 and t2 = 2.0**n2 */
  PUT_BITS_DP64(((__INT8_T)n1 + EXPBIAS_DP64) << EXPSHIFTBITS_DP64, t1);
  PUT_BITS_DP64(((__INT8_T)n2 + EXPBIAS_DP64) << EXPSHIFTBITS_DP64, t2);
  return (x * t1) * t2;
}
#endif /* USE_SCALEDOUBLE_2 */

#if defined(USE_SCALEDOUBLE_3)
/* Scales the double x by 2.0**n.
   Assumes 3*EMIN <= n <= 3*EMAX, though this condition is not checked. */
static inline double
scaleDouble_3(double x, int n)
{
  double t1, t2, t3;
  int n1, n2, n3;
  n1 = n / 3;
  n2 = (n - n1) / 2;
  n3 = n - n1 - n2;
  /* Construct the numbers t1 = 2.0**n1, t2 = 2.0**n2 and t3 = 2.0**n3 */
  PUT_BITS_DP64(((__INT8_T)n1 + EXPBIAS_DP64) << EXPSHIFTBITS_DP64, t1);
  PUT_BITS_DP64(((__INT8_T)n2 + EXPBIAS_DP64) << EXPSHIFTBITS_DP64, t2);
  PUT_BITS_DP64(((__INT8_T)n3 + EXPBIAS_DP64) << EXPSHIFTBITS_DP64, t3);
  return ((x * t1) * t2) * t3;
}
#endif /* USE_SCALEDOUBLE_3 */

#if defined(USE_SCALEFLOAT_1)
/* Scales the float x by 2.0**n.
   Assumes EMIN <= n <= EMAX, though this condition is not checked. */
static inline float
scaleFloat_1(float x, int n)
{
  float t;
  /* Construct the number t = 2.0**n */
  PUT_BITS_SP32((n + EXPBIAS_SP32) << EXPSHIFTBITS_SP32, t);
  return x * t;
}
#endif /* USE_SCALEFLOAT_1 */

#if defined(USE_SCALEFLOAT_2)
/* Scales the float x by 2.0**n.
   Assumes 2*EMIN <= n <= 2*EMAX, though this condition is not checked. */
static inline float
scaleFloat_2(float x, int n)
{
  float t1, t2;
  int n1, n2;
  n1 = n / 2;
  n2 = n - n1;
  /* Construct the numbers t1 = 2.0**n1 and t2 = 2.0**n2 */
  PUT_BITS_SP32((n1 + EXPBIAS_SP32) << EXPSHIFTBITS_SP32, t1);
  PUT_BITS_SP32((n2 + EXPBIAS_SP32) << EXPSHIFTBITS_SP32, t2);
  return (x * t1) * t2;
}
#endif /* USE_SCALEFLOAT_2 */

#if defined(USE_SCALEFLOAT_3)
/* Scales the float x by 2.0**n.
   Assumes 3*EMIN <= n <= 3*EMAX, though this condition is not checked. */
static inline float
scaleFloat_3(float x, int n)
{
  float t1, t2, t3;
  int n1, n2, n3;
  n1 = n / 3;
  n2 = (n - n1) / 2;
  n3 = n - n1 - n2;
  /* Construct the numbers t1 = 2.0**n1, t2 = 2.0**n2 and t3 = 2.0**n3 */
  PUT_BITS_SP32((n1 + EXPBIAS_SP32) << EXPSHIFTBITS_SP32, t1);
  PUT_BITS_SP32((n2 + EXPBIAS_SP32) << EXPSHIFTBITS_SP32, t2);
  PUT_BITS_SP32((n3 + EXPBIAS_SP32) << EXPSHIFTBITS_SP32, t3);
  return ((x * t1) * t2) * t3;
}
#endif /* USE_SCALEFLOAT_3 */

#if defined(USE_SETPRECISIONDOUBLE)
unsigned int
setPrecisionDouble(void)
{
  unsigned int cw, cwold = 0;
  /* There is no precision control on Hammer */
  return cwold;
}
#endif /* USE_SETPRECISIONDOUBLE */

#if defined(USE_RESTOREPRECISION)
void
restorePrecision(unsigned int cwold)
{
#if defined(linux)
/* There is no precision control on Hammer */
#elif defined(INTERIX86)
#elif defined(TARGET_OSX_X86)
#else
#error Unknown machine
#endif
  return;
}
#endif /* USE_RESTOREPRECISION */

#if defined(USE_CLEAR_FPSW_FLAGS)
/* Clears floating-point status flags. The argument should be
   the bitwise or of the flags to be cleared, from the
   list above, e.g.
     clear_fpsw_flags(AMD_F_INEXACT | AMD_F_INVALID);
 */
static inline void
clear_fpsw_flags(int flags)
{
#if defined(DONOTDEFINE_WINDOWS)
  unsigned int cw = _mm_getcsr();
  cw &= (~flags);
  _mm_setcsr(cw);
#elif defined(linux)
  unsigned int cw;
  /* Get the current floating-point control/status word */
  asm volatile("STMXCSR %0" : "=m"(cw));
  cw &= (~flags);
  asm volatile("LDMXCSR %0" : : "m"(cw));
#elif defined(INTERIX86) || defined(TARGET_WIN)
  unsigned int cw;
  /* Get the current floating-point control/status word */
  asm volatile("STMXCSR %0" : "=m"(cw));
  cw &= (~flags);
  asm volatile("LDMXCSR %0" : : "m"(cw));
#else
#error Unknown machine
#endif
}
#endif /* USE_CLEAR_FPSW_FLAGS */

#if defined(USE_RAISE_FPSW_FLAGS)
/* Raises floating-point status flags. The argument should be
   the bitwise or of the flags to be raised, from the
   list above, e.g.
     raise_fpsw_flags(AMD_F_INEXACT | AMD_F_INVALID);
 */
static inline void
raise_fpsw_flags(int flags)
{
#if defined(DONOTDEFINE_WINDOWS)
  _mm_setcsr(_mm_getcsr() | flags);
#elif defined(linux)
  unsigned int cw;
  /* Get the current floating-point control/status word */
  asm volatile("STMXCSR %0" : "=m"(cw));
  cw |= flags;
  asm volatile("LDMXCSR %0" : : "m"(cw));
#elif defined(INTERIX86) || defined(__INTERIX) || defined(TARGET_OSX_X86) ||   defined(TARGET_WIN)
  unsigned int cw;
  /* Get the current floating-point control/status word */
  asm volatile("STMXCSR %0" : "=m"(cw));
  cw |= flags;
  asm volatile("LDMXCSR %0" : : "m"(cw));
#else
#error Unknown machine
#endif
}
#endif /* USE_RAISE_FPSW_FLAGS */

#if defined(USE_GET_FPSW_INLINE)
/* Return the current floating-point status word */
static inline unsigned int
get_fpsw_inline(void)
{
#if defined(TARGET_WIN)
  return _mm_getcsr();
#elif defined(linux)
  unsigned int sw;
  asm volatile("STMXCSR %0" : "=m"(sw));
  return sw;
#elif defined(INTERIX86)
  unsigned int sw;
  asm volatile("STMXCSR %0" : "=m"(sw));
  return sw;
#else
#error Unknown machine
#endif
}
#endif /* USE_GET_FPSW_INLINE */

#if defined(USE_SET_FPSW_INLINE)
/* Set the floating-point status word */
static inline void
set_fpsw_inline(unsigned int sw)
{
#if defined(TARGET_WIN)
  _mm_setcsr(sw);
#elif defined(linux)
  /* Set the current floating-point control/status word */
  asm volatile("LDMXCSR %0" : : "m"(sw));
#elif defined(INTERIX86)
  /* Set the current floating-point control/status word */
  asm volatile("LDMXCSR %0" : : "m"(sw));
#else
#error Unknown machine
#endif
}
#endif /* USE_SET_FPSW_INLINE */

#if defined(USE_CLEAR_FPSW_INLINE)
/* Clear all exceptions from the floating-point status word */
static inline void
clear_fpsw_inline(void)
{
#if defined(TARGET_WIN)
  unsigned int cw;
  cw = _mm_getcsr();
  cw &= ~(AMD_F_INEXACT | AMD_F_UNDERFLOW | AMD_F_OVERFLOW | AMD_F_DIVBYZERO |
          AMD_F_INVALID);
  _mm_setcsr(cw);
#elif defined(linux)
  unsigned int cw;
  /* Get the current floating-point control/status word */
  asm volatile("STMXCSR %0" : "=m"(cw));
  cw &= ~(AMD_F_INEXACT | AMD_F_UNDERFLOW | AMD_F_OVERFLOW | AMD_F_DIVBYZERO |
          AMD_F_INVALID);
  asm volatile("LDMXCSR %0" : : "m"(cw));
#elif defined(INTERIX86)
  unsigned int cw;
  /* Get the current floating-point control/status word */
  asm volatile("STMXCSR %0" : "=m"(cw));
  cw &= ~(AMD_F_INEXACT | AMD_F_UNDERFLOW | AMD_F_OVERFLOW | AMD_F_DIVBYZERO |
          AMD_F_INVALID);
  asm volatile("LDMXCSR %0" : : "m"(cw));
#else
#error Unknown machine
#endif
}
#endif /* USE_CLEAR_FPSW_INLINE */

#if defined(USE_VAL_WITH_FLAGS)
/* Returns a double value after raising the given flags,
  e.g.  val_with_flags(AMD_F_INEXACT);
 */
static inline double
val_with_flags(double val, int flags)
{
  raise_fpsw_flags(flags);
  return val;
}
#endif /* USE_VAL_WITH_FLAGS */

#if defined(USE_VALF_WITH_FLAGS)
/* Returns a float value after raising the given flags,
  e.g.  valf_with_flags(AMD_F_INEXACT);
 */
static inline float
valf_with_flags(float val, int flags)
{
  raise_fpsw_flags(flags);
  return val;
}
#endif /* USE_VALF_WITH_FLAGS */

#if defined(USE_ZERO_WITH_FLAGS)
/* Returns a double +zero after raising the given flags,
  e.g.  zero_with_flags(AMD_F_INEXACT | AMD_F_INVALID);
 */
static inline double
zero_with_flags(int flags)
{
  raise_fpsw_flags(flags);
  return 0.0;
}
#endif /* USE_ZERO_WITH_FLAGS */

#if defined(USE_ZEROF_WITH_FLAGS)
/* Returns a float +zero after raising the given flags,
  e.g.  zerof_with_flags(AMD_F_INEXACT | AMD_F_INVALID);
 */
static inline float
zerof_with_flags(int flags)
{
  raise_fpsw_flags(flags);
  return 0.0F;
}
#endif /* USE_ZEROF_WITH_FLAGS */

#if defined(USE_NAN_WITH_FLAGS)
/* Returns a double quiet +nan after raising the given flags,
   e.g.  nan_with_flags(AMD_F_INVALID);
*/
static inline double
nan_with_flags(int flags)
{
  double z;
  raise_fpsw_flags(flags);
  PUT_BITS_DP64(0x7ff8000000000000, z);
  return z;
}
#endif /* USE_NAN_WITH_FLAGS */

#if defined(USE_NANF_WITH_FLAGS)
/* Returns a float quiet +nan after raising the given flags,
   e.g.  nanf_with_flags(AMD_F_INVALID);
*/
static inline float
nanf_with_flags(int flags)
{
  float z;
  raise_fpsw_flags(flags);
  PUT_BITS_SP32(0x7fc00000, z);
  return z;
}
#endif /* USE_NANF_WITH_FLAGS */

#if defined(USE_INDEFINITE_WITH_FLAGS)
/* Returns a double indefinite after raising the given flags,
   e.g.  indefinite_with_flags(AMD_F_INVALID);
*/
static inline double
indefinite_with_flags(int flags)
{
  double z;
  raise_fpsw_flags(flags);
  PUT_BITS_DP64(0xfff8000000000000, z);
  return z;
}
#endif /* USE_INDEFINITE_WITH_FLAGS */

#if defined(USE_INDEFINITEF_WITH_FLAGS)
/* Returns a float quiet +indefinite after raising the given flags,
   e.g.  indefinitef_with_flags(AMD_F_INVALID);
*/
static inline float
indefinitef_with_flags(int flags)
{
  float z;
  raise_fpsw_flags(flags);
  PUT_BITS_SP32(0xffc00000, z);
  return z;
}
#endif /* USE_INDEFINITEF_WITH_FLAGS */

#ifdef USE_INFINITY_WITH_FLAGS
/* Returns a positive double infinity after raising the given flags,
   e.g.  infinity_with_flags(AMD_F_OVERFLOW);
*/
static inline double
infinity_with_flags(int flags)
{
  double z;
  raise_fpsw_flags(flags);
  PUT_BITS_DP64((__UINT8_T)(BIASEDEMAX_DP64 + 1) << EXPSHIFTBITS_DP64, z);
  return z;
}
#endif /* USE_INFINITY_WITH_FLAGS */

#ifdef USE_INFINITYF_WITH_FLAGS
/* Returns a positive float infinity after raising the given flags,
   e.g.  infinityf_with_flags(AMD_F_OVERFLOW);
*/
static inline float
infinityf_with_flags(int flags)
{
  float z;
  raise_fpsw_flags(flags);
  PUT_BITS_SP32((BIASEDEMAX_SP32 + 1) << EXPSHIFTBITS_SP32, z);
  return z;
}
#endif /* USE_INFINITYF_WITH_FLAGS */

#if defined(USE_SPLITEXP)
/* Compute the values m, z1, and z2 such that base**x = 2**m * (z1 + z2).
   Small arguments abs(x) < 1/(16*ln(base)) and extreme arguments
   abs(x) > large/(ln(base)) (where large is the largest representable
   floating point number) should be handled separately instead of calling
   this function. This function is called by exp_amd, exp2_amd, exp10_amd,
   cosh_amd and sinh_amd. */
static inline void
splitexp(double x, double logbase, double thirtytwo_by_logbaseof2,
         double logbaseof2_by_32_lead, double logbaseof2_by_32_trail, int *m,
         double *z1, double *z2)
{
  double q, r, r1, r2, f1, f2;
  int n, j;

  /* Arrays two_to_jby32_lead_table and two_to_jby32_trail_table contain
     leading and trailing parts respectively of precomputed
     values of pow(2.0,j/32.0), for j = 0, 1, ..., 31.
     two_to_jby32_lead_table contains the first 25 bits of precision,
     and two_to_jby32_trail_table contains a further 53 bits precision. */

  static const double two_to_jby32_lead_table[32] = {
      1.00000000000000000000e+00,  /* 0x3ff0000000000000 */
      1.02189713716506958008e+00,  /* 0x3ff059b0d0000000 */
      1.04427373409271240234e+00,  /* 0x3ff0b55860000000 */
      1.06714040040969848633e+00,  /* 0x3ff11301d0000000 */
      1.09050768613815307617e+00,  /* 0x3ff172b830000000 */
      1.11438673734664916992e+00,  /* 0x3ff1d48730000000 */
      1.13878858089447021484e+00,  /* 0x3ff2387a60000000 */
      1.16372483968734741211e+00,  /* 0x3ff29e9df0000000 */
      1.18920707702636718750e+00,  /* 0x3ff306fe00000000 */
      1.21524733304977416992e+00,  /* 0x3ff371a730000000 */
      1.24185776710510253906e+00,  /* 0x3ff3dea640000000 */
      1.26905095577239990234e+00,  /* 0x3ff44e0860000000 */
      1.29683953523635864258e+00,  /* 0x3ff4bfdad0000000 */
      1.32523661851882934570e+00,  /* 0x3ff5342b50000000 */
      1.35425549745559692383e+00,  /* 0x3ff5ab07d0000000 */
      1.38390988111495971680e+00,  /* 0x3ff6247eb0000000 */
      1.41421353816986083984e+00,  /* 0x3ff6a09e60000000 */
      1.44518077373504638672e+00,  /* 0x3ff71f75e0000000 */
      1.47682613134384155273e+00,  /* 0x3ff7a11470000000 */
      1.50916439294815063477e+00,  /* 0x3ff8258990000000 */
      1.54221081733703613281e+00,  /* 0x3ff8ace540000000 */
      1.57598084211349487305e+00,  /* 0x3ff93737b0000000 */
      1.61049032211303710938e+00,  /* 0x3ff9c49180000000 */
      1.64575546979904174805e+00,  /* 0x3ffa5503b0000000 */
      1.68179279565811157227e+00,  /* 0x3ffae89f90000000 */
      1.71861928701400756836e+00,  /* 0x3ffb7f76f0000000 */
      1.75625211000442504883e+00,  /* 0x3ffc199bd0000000 */
      1.79470902681350708008e+00,  /* 0x3ffcb720d0000000 */
      1.83400803804397583008e+00,  /* 0x3ffd5818d0000000 */
      1.87416762113571166992e+00,  /* 0x3ffdfc9730000000 */
      1.91520655155181884766e+00,  /* 0x3ffea4afa0000000 */
      1.95714408159255981445e+00}; /* 0x3fff507650000000 */

  static const double two_to_jby32_trail_table[32] = {
      0.00000000000000000000e+00,  /* 0x0000000000000000 */
      1.14890470981563546737e-08,  /* 0x3e48ac2ba1d73e2a */
      4.83347014379782142328e-08,  /* 0x3e69f3121ec53172 */
      2.67125131841396124714e-10,  /* 0x3df25b50a4ebbf1b */
      4.65271045830351350190e-08,  /* 0x3e68faa2f5b9bef9 */
      5.24924336638693782574e-09,  /* 0x3e368b9aa7805b80 */
      5.38622214388600821910e-08,  /* 0x3e6ceac470cd83f6 */
      1.90902301017041969782e-08,  /* 0x3e547f7b84b09745 */
      3.79763538792174980894e-08,  /* 0x3e64636e2a5bd1ab */
      2.69306947081946450986e-08,  /* 0x3e5ceaa72a9c5154 */
      4.49683815095311756138e-08,  /* 0x3e682468446b6824 */
      1.41933332021066904914e-09,  /* 0x3e18624b40c4dbd0 */
      1.94146510233556266402e-08,  /* 0x3e54d8a89c750e5e */
      2.46409119489264118569e-08,  /* 0x3e5a753e077c2a0f */
      4.94812958044698886494e-08,  /* 0x3e6a90a852b19260 */
      8.48872238075784476136e-10,  /* 0x3e0d2ac258f87d03 */
      2.42032342089579394887e-08,  /* 0x3e59fcef32422cbf */
      3.32420002333182569170e-08,  /* 0x3e61d8bee7ba46e2 */
      1.45956577586525322754e-08,  /* 0x3e4f580c36bea881 */
      3.46452721050003920866e-08,  /* 0x3e62999c25159f11 */
      8.07090469079979051284e-09,  /* 0x3e415506dadd3e2a */
      2.99439161340839520436e-09,  /* 0x3e29b8bc9e8a0388 */
      9.83621719880452147153e-09,  /* 0x3e451f8480e3e236 */
      8.35492309647188080486e-09,  /* 0x3e41f12ae45a1224 */
      3.48493175137966283582e-08,  /* 0x3e62b5a75abd0e6a */
      1.11084703472699692902e-08,  /* 0x3e47daf237553d84 */
      5.03688744342840346564e-08,  /* 0x3e6b0aa538444196 */
      4.81896001063495806249e-08,  /* 0x3e69df20d22a0798 */
      4.83653666334089557746e-08,  /* 0x3e69f7490e4bb40b */
      1.29745882314081237628e-08,  /* 0x3e4bdcdaf5cb4656 */
      9.84532844621636118964e-09,  /* 0x3e452486cc2c7b9d */
      4.25828404545651943883e-08}; /* 0x3e66dc8a80ce9f09 */

  /*
    Step 1. Reduce the argument.

    To perform argument reduction, we find the integer n such that
    x = n * logbaseof2/32 + remainder, |remainder| <= logbaseof2/64.
    n is defined by round-to-nearest-integer( x*32/logbaseof2 ) and
    remainder by x - n*logbaseof2/32. The calculation of n is
    straightforward whereas the computation of x - n*logbaseof2/32
    must be carried out carefully.
    logbaseof2/32 is so represented in two pieces that
    (1) logbaseof2/32 is known to extra precision, (2) the product
    of n and the leading piece is a model number and is hence
    calculated without error, and (3) the subtraction of the value
    obtained in (2) from x is a model number and is hence again
    obtained without error.
  */

  r = x * thirtytwo_by_logbaseof2;
  /* Set n = nearest integer to r */
  /* This is faster on Hammer */
  if (r > 0)
    n = (int)(r + 0.5);
  else
    n = (int)(r - 0.5);

  r1 = x - n * logbaseof2_by_32_lead;
  r2 = -n * logbaseof2_by_32_trail;

  /* Set j = n mod 32:   5 mod 32 = 5,   -5 mod 32 = 27,  etc. */
  /* j = n % 32;
     if (j < 0) j += 32; */
  j = n & 0x0000001f;

  f1 = two_to_jby32_lead_table[j];
  f2 = two_to_jby32_trail_table[j];

  *m = (n - j) / 32;

  /* Step 2. The following is the core approximation. We approximate
     exp(r1+r2)-1 by a polynomial. */

  r1 *= logbase;
  r2 *= logbase;

  r = r1 + r2;
  q = r1 + (r2 +
            r * r * (5.00000000000000008883e-01 +
                     r * (1.66666666665260878863e-01 +
                          r * (4.16666666662260795726e-02 +
                               r * (8.33336798434219616221e-03 +
                                    r * (1.38889490863777199667e-03))))));

  /* Step 3. Function value reconstruction.
     We now reconstruct the exponential of the input argument
     so that exp(x) = 2**m * (z1 + z2).
     The order of the computation below must be strictly observed. */

  *z1 = f1;
  *z2 = f2 + ((f1 + f2) * q);
}
#endif /* USE_SPLITEXP */

#if defined(USE_SPLITEXPF)
/* Compute the values m, z1, and z2 such that base**x = 2**m * (z1 + z2).
   Small arguments abs(x) < 1/(16*ln(base)) and extreme arguments
   abs(x) > large/(ln(base)) (where large is the largest representable
   floating point number) should be handled separately instead of calling
   this function. This function is called by exp_amd, exp2_amd, exp10_amd,
   cosh_amd and sinh_amd. */
static inline void
splitexpf(float x, float logbase, float thirtytwo_by_logbaseof2,
          float logbaseof2_by_32_lead, float logbaseof2_by_32_trail, int *m,
          float *z1, float *z2)
{
  float q, r, r1, r2, f1, f2;
  int n, j;

  /* Arrays two_to_jby32_lead_table and two_to_jby32_trail_table contain
     leading and trailing parts respectively of precomputed
     values of pow(2.0,j/32.0), for j = 0, 1, ..., 31.
     two_to_jby32_lead_table contains the first 10 bits of precision,
     and two_to_jby32_trail_table contains a further 24 bits precision. */

  static const float two_to_jby32_lead_table[32] = {
      1.0000000000E+00F,  /* 0x3F800000 */
      1.0214843750E+00F,  /* 0x3F82C000 */
      1.0429687500E+00F,  /* 0x3F858000 */
      1.0664062500E+00F,  /* 0x3F888000 */
      1.0898437500E+00F,  /* 0x3F8B8000 */
      1.1132812500E+00F,  /* 0x3F8E8000 */
      1.1386718750E+00F,  /* 0x3F91C000 */
      1.1621093750E+00F,  /* 0x3F94C000 */
      1.1875000000E+00F,  /* 0x3F980000 */
      1.2148437500E+00F,  /* 0x3F9B8000 */
      1.2402343750E+00F,  /* 0x3F9EC000 */
      1.2675781250E+00F,  /* 0x3FA24000 */
      1.2949218750E+00F,  /* 0x3FA5C000 */
      1.3242187500E+00F,  /* 0x3FA98000 */
      1.3535156250E+00F,  /* 0x3FAD4000 */
      1.3828125000E+00F,  /* 0x3FB10000 */
      1.4140625000E+00F,  /* 0x3FB50000 */
      1.4433593750E+00F,  /* 0x3FB8C000 */
      1.4765625000E+00F,  /* 0x3FBD0000 */
      1.5078125000E+00F,  /* 0x3FC10000 */
      1.5410156250E+00F,  /* 0x3FC54000 */
      1.5742187500E+00F,  /* 0x3FC98000 */
      1.6093750000E+00F,  /* 0x3FCE0000 */
      1.6445312500E+00F,  /* 0x3FD28000 */
      1.6816406250E+00F,  /* 0x3FD74000 */
      1.7167968750E+00F,  /* 0x3FDBC000 */
      1.7558593750E+00F,  /* 0x3FE0C000 */
      1.7929687500E+00F,  /* 0x3FE58000 */
      1.8339843750E+00F,  /* 0x3FEAC000 */
      1.8730468750E+00F,  /* 0x3FEFC000 */
      1.9140625000E+00F,  /* 0x3FF50000 */
      1.9570312500E+00F}; /* 0x3FFA8000 */

  static const float two_to_jby32_trail_table[32] = {
      0.0000000000E+00F,  /* 0x00000000 */
      4.1277357377E-04F,  /* 0x39D86988 */
      1.3050324051E-03F,  /* 0x3AAB0D9F */
      7.3415064253E-04F,  /* 0x3A407404 */
      6.6398258787E-04F,  /* 0x3A2E0F1E */
      1.1054925853E-03F,  /* 0x3A90E62D */
      1.1675967835E-04F,  /* 0x38F4DCE0 */
      1.6154836630E-03F,  /* 0x3AD3BEA3 */
      1.7071149778E-03F,  /* 0x3ADFC146 */
      4.0360994171E-04F,  /* 0x39D39B9C */
      1.6234370414E-03F,  /* 0x3AD4C982 */
      1.4728321694E-03F,  /* 0x3AC10C0C */
      1.9176795613E-03F,  /* 0x3AFB5AA6 */
      1.0178930825E-03F,  /* 0x3A856AD3 */
      7.3992193211E-04F,  /* 0x3A41F752 */
      1.0973819299E-03F,  /* 0x3A8FD607 */
      1.5106226783E-04F,  /* 0x391E6678 */
      1.8214319134E-03F,  /* 0x3AEEBD1D */
      2.6364589576E-04F,  /* 0x398A39F4 */
      1.3519275235E-03F,  /* 0x3AB13329 */
      1.1952003697E-03F,  /* 0x3A9CA845 */
      1.7620950239E-03F,  /* 0x3AE6F619 */
      1.1153318919E-03F,  /* 0x3A923054 */
      1.2242280645E-03F,  /* 0x3AA07647 */
      1.5220546629E-04F,  /* 0x391F9958 */
      1.8224230735E-03F,  /* 0x3AEEDE5F */
      3.9278529584E-04F,  /* 0x39CDEEC0 */
      1.7403248930E-03F,  /* 0x3AE41B9D */
      2.3711356334E-05F,  /* 0x37C6E7C0 */
      1.1207590578E-03F,  /* 0x3A92E66F */
      1.1440613307E-03F,  /* 0x3A95F454 */
      1.1287408415E-04F}; /* 0x38ECB6D0 */

  /*
    Step 1. Reduce the argument.

    To perform argument reduction, we find the integer n such that
    x = n * logbaseof2/32 + remainder, |remainder| <= logbaseof2/64.
    n is defined by round-to-nearest-integer( x*32/logbaseof2 ) and
    remainder by x - n*logbaseof2/32. The calculation of n is
    straightforward whereas the computation of x - n*logbaseof2/32
    must be carried out carefully.
    logbaseof2/32 is so represented in two pieces that
    (1) logbaseof2/32 is known to extra precision, (2) the product
    of n and the leading piece is a model number and is hence
    calculated without error, and (3) the subtraction of the value
    obtained in (2) from x is a model number and is hence again
    obtained without error.
  */

  r = x * thirtytwo_by_logbaseof2;
  /* Set n = nearest integer to r */
  /* This is faster on Hammer */
  if (r > 0)
    n = (int)(r + 0.5F);
  else
    n = (int)(r - 0.5F);

  r1 = x - n * logbaseof2_by_32_lead;
  r2 = -n * logbaseof2_by_32_trail;

  /* Set j = n mod 32:   5 mod 32 = 5,   -5 mod 32 = 27,  etc. */
  /* j = n % 32;
     if (j < 0) j += 32; */
  j = n & 0x0000001f;

  f1 = two_to_jby32_lead_table[j];
  f2 = two_to_jby32_trail_table[j];

  *m = (n - j) / 32;

  /* Step 2. The following is the core approximation. We approximate
     exp(r1+r2)-1 by a polynomial. */

  r1 *= logbase;
  r2 *= logbase;

  r = r1 + r2;
  q = r1 + (r2 +
            r * r * (5.00000000000000008883e-01F +
                     r * (1.66666666665260878863e-01F)));

  /* Step 3. Function value reconstruction.
     We now reconstruct the exponential of the input argument
     so that exp(x) = 2**m * (z1 + z2).
     The order of the computation below must be strictly observed. */

  *z1 = f1;
  *z2 = f2 + ((f1 + f2) * q);
}
#endif /* SPLITEXPF */

#if defined(USE_SCALEUPDOUBLE1024)
/* Scales up a double (normal or denormal) whose bit pattern is given
   as ux by 2**1024. There are no checks that the input number is
   scalable by that amount. */
static inline void
scaleUpDouble1024(__UINT8_T ux, __UINT8_T *ur)
{
  __UINT8_T uy;
  double y;

  if ((ux & EXPBITS_DP64) == 0) {
    /* ux is denormalised */
    PUT_BITS_DP64(ux | 0x4010000000000000, y);
    if (ux & SIGNBIT_DP64)
      y += 4.0;
    else
      y -= 4.0;
    GET_BITS_DP64(y, uy);
  } else
    /* ux is normal */
    uy = ux + 0x4000000000000000;

  *ur = uy;
  return;
}

#endif /* SCALEUPDOUBLE1024 */

#if defined(USE_SCALEDOWNDOUBLE)
/* Scales down a double whose bit pattern is given as ux by 2**k.
   There are no checks that the input number is scalable by that amount. */
static inline void
scaleDownDouble(__UINT8_T ux, int k, __UINT8_T *ur)
{
  __UINT8_T uy, uk, ax, xsign;
  int n, shift;
  xsign = ux & SIGNBIT_DP64;
  ax = ux & ~SIGNBIT_DP64;
  n = (int)((ax & EXPBITS_DP64) >> EXPSHIFTBITS_DP64) - k;
  if (n > 0) {
    uk = (__UINT8_T)n << EXPSHIFTBITS_DP64;
    uy = (ax & ~EXPBITS_DP64) | uk;
  } else {
    uy = (ax & ~EXPBITS_DP64) | 0x0010000000000000;
    shift = (1 - n);
    if (shift > MANTLENGTH_DP64 + 1)
      /* Sigh. Shifting works mod 64 so be careful not to shift too much */
      uy = 0;
    else {
      /* Make sure we round the result */
      uy >>= shift - 1;
      uy = (uy >> 1) + (uy & 1);
    }
  }
  *ur = uy | xsign;
}

#endif /* SCALEDOWNDOUBLE */

#if defined(USE_SCALEUPFLOAT128)
/* Scales up a float (normal or denormal) whose bit pattern is given
   as ux by 2**128. There are no checks that the input number is
   scalable by that amount. */
static inline void
scaleUpFloat128(unsigned int ux, unsigned int *ur)
{
  unsigned int uy;
  float y;

  if ((ux & EXPBITS_SP32) == 0) {
    /* ux is denormalised */
    PUT_BITS_SP32(ux | 0x40800000, y);
    /* Compensate for the implicit bit just added */
    if (ux & SIGNBIT_SP32)
      y += 4.0F;
    else
      y -= 4.0F;
    GET_BITS_SP32(y, uy);
  } else
    /* ux is normal */
    uy = ux + 0x40000000;
  *ur = uy;
}
#endif /* SCALEUPFLOAT128 */

#if defined(USE_SCALEDOWNFLOAT)
/* Scales down a float whose bit pattern is given as ux by 2**k.
   There are no checks that the input number is scalable by that amount. */
static inline void
scaleDownFloat(unsigned int ux, int k, unsigned int *ur)
{
  unsigned int uy, uk, ax, xsign;
  int n, shift;

  xsign = ux & SIGNBIT_SP32;
  ax = ux & ~SIGNBIT_SP32;
  n = ((ax & EXPBITS_SP32) >> EXPSHIFTBITS_SP32) - k;
  if (n > 0) {
    uk = (unsigned int)n << EXPSHIFTBITS_SP32;
    uy = (ax & ~EXPBITS_SP32) | uk;
  } else {
    uy = (ax & ~EXPBITS_SP32) | 0x00800000;
    shift = (1 - n);
    if (shift > MANTLENGTH_SP32 + 1)
      /* Sigh. Shifting works mod 32 so be careful not to shift too much */
      uy = 0;
    else {
      /* Make sure we round the result */
      uy >>= shift - 1;
      uy = (uy >> 1) + (uy & 1);
    }
  }
  *ur = uy | xsign;
}
#endif /* SCALEDOWNFLOAT */

#if defined(USE_SQRT_AMD_INLINE)
static inline double
sqrt_amd_inline(double x)
{
  /*
     Computes the square root of x.

     The calculation is carried out in three steps.

     Step 1. Reduction.
     The input argument is scaled to the interval [1, 4) by
     computing
               x = 2^e * y, where y in [1,4).
     Furthermore y is decomposed as y = c + t where
               c = 1 + j/32, j = 0,1,..,96; and |t| <= 1/64.

     Step 2. Approximation.
     An approximation q = sqrt(1 + (t/c)) - 1  is obtained
     from a basic series expansion using precomputed values
     stored in rt_jby32_lead_table_dbl and rt_jby32_trail_table_dbl.

     Step 3. Reconstruction.
     The value of sqrt(x) is reconstructed via
       sqrt(x) = 2^(e/2) * sqrt(y)
               = 2^(e/2) * sqrt(c) * sqrt(y/c)
               = 2^(e/2) * sqrt(c) * sqrt(1 + t/c)
               = 2^(e/2) * [ sqrt(c) + sqrt(c)*q ]
    */

  __UINT8_T ux, ax, u;
  double r1, r2, c, y, p, q, r, twop, z, rtc, rtc_lead, rtc_trail;
  int e, denorm = 0, index;

  /* Arrays rt_jby32_lead_table_dbl and rt_jby32_trail_table_dbl contain
     leading and trailing parts respectively of precomputed
     values of sqrt(j/32), for j = 32, 33, ..., 128.
     rt_jby32_lead_table_dbl contains the first 21 bits of precision,
     and rt_jby32_trail_table_dbl contains a further 53 bits precision. */

  static const double rt_jby32_lead_table_dbl[97] = {
      1.00000000000000000000e+00,  /* 0x3ff0000000000000 */
      1.01550388336181640625e+00,  /* 0x3ff03f8100000000 */
      1.03077602386474609375e+00,  /* 0x3ff07e0f00000000 */
      1.04582500457763671875e+00,  /* 0x3ff0bbb300000000 */
      1.06065940856933593750e+00,  /* 0x3ff0f87600000000 */
      1.07528972625732421875e+00,  /* 0x3ff1346300000000 */
      1.08972454071044921875e+00,  /* 0x3ff16f8300000000 */
      1.10396957397460937500e+00,  /* 0x3ff1a9dc00000000 */
      1.11803340911865234375e+00,  /* 0x3ff1e37700000000 */
      1.13192272186279296875e+00,  /* 0x3ff21c5b00000000 */
      1.14564323425292968750e+00,  /* 0x3ff2548e00000000 */
      1.15920162200927734375e+00,  /* 0x3ff28c1700000000 */
      1.17260360717773437500e+00,  /* 0x3ff2c2fc00000000 */
      1.18585395812988281250e+00,  /* 0x3ff2f94200000000 */
      1.19895744323730468750e+00,  /* 0x3ff32eee00000000 */
      1.21191978454589843750e+00,  /* 0x3ff3640600000000 */
      1.22474479675292968750e+00,  /* 0x3ff3988e00000000 */
      1.23743629455566406250e+00,  /* 0x3ff3cc8a00000000 */
      1.25000000000000000000e+00,  /* 0x3ff4000000000000 */
      1.26243782043457031250e+00,  /* 0x3ff432f200000000 */
      1.27475452423095703125e+00,  /* 0x3ff4656500000000 */
      1.28695297241210937500e+00,  /* 0x3ff4975c00000000 */
      1.29903793334960937500e+00,  /* 0x3ff4c8dc00000000 */
      1.31101036071777343750e+00,  /* 0x3ff4f9e600000000 */
      1.32287502288818359375e+00,  /* 0x3ff52a7f00000000 */
      1.33463478088378906250e+00,  /* 0x3ff55aaa00000000 */
      1.34629058837890625000e+00,  /* 0x3ff58a6800000000 */
      1.35784721374511718750e+00,  /* 0x3ff5b9be00000000 */
      1.36930561065673828125e+00,  /* 0x3ff5e8ad00000000 */
      1.38066959381103515625e+00,  /* 0x3ff6173900000000 */
      1.39194107055664062500e+00,  /* 0x3ff6456400000000 */
      1.40312099456787109375e+00,  /* 0x3ff6732f00000000 */
      1.41421318054199218750e+00,  /* 0x3ff6a09e00000000 */
      1.42521858215332031250e+00,  /* 0x3ff6cdb200000000 */
      1.43614006042480468750e+00,  /* 0x3ff6fa6e00000000 */
      1.44697952270507812500e+00,  /* 0x3ff726d400000000 */
      1.45773792266845703125e+00,  /* 0x3ff752e500000000 */
      1.46841716766357421875e+00,  /* 0x3ff77ea300000000 */
      1.47901916503906250000e+00,  /* 0x3ff7aa1000000000 */
      1.48954677581787109375e+00,  /* 0x3ff7d52f00000000 */
      1.50000000000000000000e+00,  /* 0x3ff8000000000000 */
      1.51038074493408203125e+00,  /* 0x3ff82a8500000000 */
      1.52068996429443359375e+00,  /* 0x3ff854bf00000000 */
      1.53093051910400390625e+00,  /* 0x3ff87eb100000000 */
      1.54110336303710937500e+00,  /* 0x3ff8a85c00000000 */
      1.55120849609375000000e+00,  /* 0x3ff8d1c000000000 */
      1.56124877929687500000e+00,  /* 0x3ff8fae000000000 */
      1.57122516632080078125e+00,  /* 0x3ff923bd00000000 */
      1.58113861083984375000e+00,  /* 0x3ff94c5800000000 */
      1.59099006652832031250e+00,  /* 0x3ff974b200000000 */
      1.60078048706054687500e+00,  /* 0x3ff99ccc00000000 */
      1.61051177978515625000e+00,  /* 0x3ff9c4a800000000 */
      1.62018489837646484375e+00,  /* 0x3ff9ec4700000000 */
      1.62979984283447265625e+00,  /* 0x3ffa13a900000000 */
      1.63935947418212890625e+00,  /* 0x3ffa3ad100000000 */
      1.64886283874511718750e+00,  /* 0x3ffa61be00000000 */
      1.65831184387207031250e+00,  /* 0x3ffa887200000000 */
      1.66770744323730468750e+00,  /* 0x3ffaaeee00000000 */
      1.67705059051513671875e+00,  /* 0x3ffad53300000000 */
      1.68634128570556640625e+00,  /* 0x3ffafb4100000000 */
      1.69558238983154296875e+00,  /* 0x3ffb211b00000000 */
      1.70477199554443359375e+00,  /* 0x3ffb46bf00000000 */
      1.71391296386718750000e+00,  /* 0x3ffb6c3000000000 */
      1.72300529479980468750e+00,  /* 0x3ffb916e00000000 */
      1.73204994201660156250e+00,  /* 0x3ffbb67a00000000 */
      1.74104785919189453125e+00,  /* 0x3ffbdb5500000000 */
      1.75000000000000000000e+00,  /* 0x3ffc000000000000 */
      1.75890541076660156250e+00,  /* 0x3ffc247a00000000 */
      1.76776695251464843750e+00,  /* 0x3ffc48c600000000 */
      1.77658367156982421875e+00,  /* 0x3ffc6ce300000000 */
      1.78535652160644531250e+00,  /* 0x3ffc90d200000000 */
      1.79408740997314453125e+00,  /* 0x3ffcb49500000000 */
      1.80277538299560546875e+00,  /* 0x3ffcd82b00000000 */
      1.81142139434814453125e+00,  /* 0x3ffcfb9500000000 */
      1.82002735137939453125e+00,  /* 0x3ffd1ed500000000 */
      1.82859230041503906250e+00,  /* 0x3ffd41ea00000000 */
      1.83711719512939453125e+00,  /* 0x3ffd64d500000000 */
      1.84560203552246093750e+00,  /* 0x3ffd879600000000 */
      1.85404872894287109375e+00,  /* 0x3ffdaa2f00000000 */
      1.86245727539062500000e+00,  /* 0x3ffdcca000000000 */
      1.87082862854003906250e+00,  /* 0x3ffdeeea00000000 */
      1.87916183471679687500e+00,  /* 0x3ffe110c00000000 */
      1.88745784759521484375e+00,  /* 0x3ffe330700000000 */
      1.89571857452392578125e+00,  /* 0x3ffe54dd00000000 */
      1.90394306182861328125e+00,  /* 0x3ffe768d00000000 */
      1.91213226318359375000e+00,  /* 0x3ffe981800000000 */
      1.92028617858886718750e+00,  /* 0x3ffeb97e00000000 */
      1.92840576171875000000e+00,  /* 0x3ffedac000000000 */
      1.93649101257324218750e+00,  /* 0x3ffefbde00000000 */
      1.94454288482666015625e+00,  /* 0x3fff1cd900000000 */
      1.95256233215332031250e+00,  /* 0x3fff3db200000000 */
      1.96054744720458984375e+00,  /* 0x3fff5e6700000000 */
      1.96850109100341796875e+00,  /* 0x3fff7efb00000000 */
      1.97642326354980468750e+00,  /* 0x3fff9f6e00000000 */
      1.98431301116943359375e+00,  /* 0x3fffbfbf00000000 */
      1.99217128753662109375e+00,  /* 0x3fffdfef00000000 */
      2.00000000000000000000e+00}; /* 0x4000000000000000 */

  static const double rt_jby32_trail_table_dbl[97] = {
      0.00000000000000000000e+00,  /* 0x0000000000000000 */
      9.17217678638807524014e-07,  /* 0x3eaec6d70177881c */
      3.82539669043705364790e-07,  /* 0x3e99abfb41bd6b24 */
      2.85899577162227138140e-08,  /* 0x3e5eb2bf6bab55a2 */
      7.63210485349101216659e-07,  /* 0x3ea99bed9b2d8d0c */
      9.32123004127716212874e-07,  /* 0x3eaf46e029c1b296 */
      1.95174719169309219157e-07,  /* 0x3e8a3226fc42f30c */
      5.34316371481845492427e-07,  /* 0x3ea1edbe20701d73 */
      5.79631242504454563052e-07,  /* 0x3ea372fe94f82be7 */
      4.20404384109571705948e-07,  /* 0x3e9c367e08e7bb06 */
      6.89486030314147010716e-07,  /* 0x3ea722a3d0a66608 */
      6.89927685625314560328e-07,  /* 0x3ea7266f067ca1d6 */
      3.32778123013641425828e-07,  /* 0x3e965515a9b34850 */
      1.64433259436999584387e-07,  /* 0x3e8611e23ef6c1bd */
      4.37590875197899335723e-07,  /* 0x3e9d5dc1059ed8e7 */
      1.79808183816018617413e-07,  /* 0x3e88222982d0e4f4 */
      7.46386593615986477624e-08,  /* 0x3e7409212e7d0322 */
      5.72520794105201454728e-07,  /* 0x3ea335ea8a5fcf39 */
      0.00000000000000000000e+00,  /* 0x0000000000000000 */
      2.96860689431670420344e-07,  /* 0x3e93ec071e938bfe */
      3.54167239176257065345e-07,  /* 0x3e97c48bfd9862c6 */
      7.95211265664474710063e-07,  /* 0x3eaaaed010f74671 */
      1.72327048595145565621e-07,  /* 0x3e87211cbfeb62e0 */
      6.99494915996239297020e-07,  /* 0x3ea7789d9660e72d */
      6.32644111701500844315e-07,  /* 0x3ea53a5f1d36f1cf */
      6.20124838851440463844e-10,  /* 0x3e054eacff2057dc */
      6.13404719757812629969e-07,  /* 0x3ea4951b3e6a83cc */
      3.47654909777986407387e-07,  /* 0x3e9754aa76884c66 */
      7.83106177002392475763e-07,  /* 0x3eaa46d4b1de1074 */
      5.33337372440526357008e-07,  /* 0x3ea1e55548f92635 */
      2.01508648555298681765e-08,  /* 0x3e55a3070dd17788 */
      5.25472356925843939587e-07,  /* 0x3ea1a1c5eedb0801 */
      3.81831102861301692797e-07,  /* 0x3e999fcef32422cc */
      6.99220602161420018738e-07,  /* 0x3ea776425d6b0199 */
      6.01209702477462624811e-07,  /* 0x3ea42c5a1e0191a2 */
      9.01437000591944740554e-08,  /* 0x3e7832a0bdff1327 */
      5.10428680864685379950e-08,  /* 0x3e6b674743636676 */
      3.47895267104621031421e-07,  /* 0x3e9758cb90d2f714 */
      7.80735841510641848628e-07,  /* 0x3eaa3278459cde25 */
      1.35158752025506517690e-07,  /* 0x3e822404f4a103ee */
      0.00000000000000000000e+00,  /* 0x0000000000000000 */
      1.76523947728535489812e-09,  /* 0x3e1e539af6892ac5 */
      6.68280121328499932183e-07,  /* 0x3ea66c7b872c9cd0 */
      5.70135482405123276616e-07,  /* 0x3ea3216d2f43887d */
      1.37705134737562525897e-07,  /* 0x3e827b832cbedc0e */
      7.09655107074516613672e-07,  /* 0x3ea7cfe41579091d */
      7.20302724551461693011e-07,  /* 0x3ea82b5a713c490a */
      4.69926266058212796694e-07,  /* 0x3e9f8945932d872e */
      2.19244345915999437026e-07,  /* 0x3e8d6d2da9490251 */
      1.91141411617401877927e-07,  /* 0x3e89a791a3114e4a */
      5.72297665296622053774e-07,  /* 0x3ea333ffe005988d */
      5.61055484436830560103e-07,  /* 0x3ea2d36e0ed49ab1 */
      2.76225500213991506100e-07,  /* 0x3e92898498f55f9e */
      7.58466189522395692908e-07,  /* 0x3ea9732cca1032a3 */
      1.56893371256836029827e-07,  /* 0x3e850ed0b02a22d2 */
      4.06038997708867066507e-07,  /* 0x3e9b3fb265b1e40a */
      5.51305629612057435809e-07,  /* 0x3ea27fade682d1de */
      5.64778487026561123207e-07,  /* 0x3ea2f36906f707ba */
      3.92609705553556897517e-07,  /* 0x3e9a58fbbee883b6 */
      9.09698438776943827802e-07,  /* 0x3eae864005bca6d7 */
      1.05949774066016139743e-07,  /* 0x3e7c70d02300f263 */
      7.16578798392844784244e-07,  /* 0x3ea80b5d712d8e3e */
      6.86233073531233972561e-07,  /* 0x3ea706b27cc7d390 */
      7.99211473033494452908e-07,  /* 0x3eaad12c9d849a97 */
      8.65552275731027456121e-07,  /* 0x3ead0b09954e764b */
      6.75456120386058448618e-07,  /* 0x3ea6aa1fb7826cbd */
      0.00000000000000000000e+00,  /* 0x0000000000000000 */
      4.99167184520462138743e-07,  /* 0x3ea0bfd03f46763c */
      4.51720373502110930296e-10,  /* 0x3dff0abfb4adfb9e */
      1.28874162718371367439e-07,  /* 0x3e814c151f991b2e */
      5.85529267186999798656e-07,  /* 0x3ea3a5a879b09292 */
      1.01827770937125531924e-07,  /* 0x3e7b558d173f9796 */
      2.54736389177809626508e-07,  /* 0x3e9118567cd83fb8 */
      6.98925535290464831294e-07,  /* 0x3ea773b981896751 */
      1.20940735036524314513e-07,  /* 0x3e803b7df49f48a8 */
      5.43759351196479689657e-08,  /* 0x3e6d315f22491900 */
      1.11957989042397958409e-07,  /* 0x3e7e0db1c5bb84b2 */
      8.47006714134442661218e-07,  /* 0x3eac6bbb7644ff76 */
      8.92831044643427836228e-07,  /* 0x3eadf55c3afec01f */
      7.77828292464916501663e-07,  /* 0x3eaa197e81034da3 */
      6.48469316302918797451e-08,  /* 0x3e71683f4920555d */
      2.12579816658859849140e-07,  /* 0x3e8c882fd78bb0b0 */
      7.61222472580559138435e-07,  /* 0x3ea98ad9eb7b83ec */
      2.86488961857314189607e-07,  /* 0x3e9339d7c7777273 */
      2.14637363790165363515e-07,  /* 0x3e8ccee237cae6fe */
      5.44137005612605847831e-08,  /* 0x3e6d368fe324a146 */
      2.58378284856442408413e-07,  /* 0x3e9156e7b6d99b45 */
      3.15848939061134843091e-07,  /* 0x3e95323e5310b5c1 */
      6.60530466255089632309e-07,  /* 0x3ea629e9db362f5d */
      7.63436345535852301127e-07,  /* 0x3ea99dde4728d7ec */
      8.68233432860324345268e-08,  /* 0x3e774e746878544d */
      9.45465175398023087082e-07,  /* 0x3eafb97be873a87d */
      8.77499534786171267246e-07,  /* 0x3ead71a9e23c2f63 */
      2.74055432394999316135e-07,  /* 0x3e92643c89cda173 */
      4.72129009349126213532e-07,  /* 0x3e9faf1d57a4d56c */
      8.93777032327078947306e-07,  /* 0x3eadfd7c7ab7b282 */
      0.00000000000000000000e+00}; /* 0x0000000000000000 */

  /* Handle special arguments first */

  GET_BITS_DP64(x, ux);
  ax = ux & (~SIGNBIT_DP64);

  if (ax >= 0x7ff0000000000000) {
    /* x is either NaN or infinity */
    if (ux & MANTBITS_DP64)
      /* x is NaN */
      return x + x; /* Raise invalid if it is a signalling NaN */
    else if (ux & SIGNBIT_DP64)
      /* x is negative infinity */
      return nan_with_flags(AMD_F_INVALID);
    else
      /* x is positive infinity */
      return x;
  } else if (ux & SIGNBIT_DP64) {
    /* x is negative. */
    if (ux == SIGNBIT_DP64)
      /* Handle negative zero first */
      return x;
    else
      return nan_with_flags(AMD_F_INVALID);
  } else if (ux <= 0x000fffffffffffff) {
    /* x is denormalised or zero */
    if (ux == 0)
      /* x is zero */
      return x;
    else {
      /* x is denormalised; scale it up */
      /* Normalize x by increasing the exponent by 60
         and subtracting a correction to account for the implicit
         bit. This replaces a slow denormalized
         multiplication by a fast normal subtraction. */
      static const double corr =
          2.5653355008114851558350183e-290; /* 0x03d0000000000000 */
      denorm = 1;
      GET_BITS_DP64(x, ux);
      PUT_BITS_DP64(ux | 0x03d0000000000000, x);
      x -= corr;
      GET_BITS_DP64(x, ux);
    }
  }

  /* Main algorithm */

  /*
     Find y and e such that x = 2^e * y, where y in [1,4).
     This is done using an in-lined variant of splitDouble,
     which also ensures that e is even.
   */
  y = x;
  ux &= EXPBITS_DP64;
  ux >>= EXPSHIFTBITS_DP64;
  if (ux & 1) {
    GET_BITS_DP64(y, u);
    u &= (SIGNBIT_DP64 | MANTBITS_DP64);
    u |= ONEEXPBITS_DP64;
    PUT_BITS_DP64(u, y);
    e = ux - EXPBIAS_DP64;
  } else {
    GET_BITS_DP64(y, u);
    u &= (SIGNBIT_DP64 | MANTBITS_DP64);
    u |= TWOEXPBITS_DP64;
    PUT_BITS_DP64(u, y);
    e = ux - EXPBIAS_DP64 - 1;
  }

  /* Find the index of the sub-interval of [1,4) in which y lies. */

  index = (int)(32.0 * y + 0.5);

  /* Look up the table values and compute c and r = c/t */

  rtc_lead = rt_jby32_lead_table_dbl[index - 32];
  rtc_trail = rt_jby32_trail_table_dbl[index - 32];
  c = 0.03125 * index;
  r = (y - c) / c;

  /*
    Find q = sqrt(1+r) - 1.
    From one step of Newton on (q+1)^2 = 1+r
  */

  p = r * 0.5 - r * r * (0.1250079870 - r * (0.6250522999E-01));
  twop = p + p;
  q = p - (p * p + (twop - r)) / (twop + 2.0);

  /* Reconstruction */

  rtc = rtc_lead + rtc_trail;
  e >>= 1; /* e = e/2 */
  z = rtc_lead + (rtc * q + rtc_trail);

  if (denorm) {
    /* Scale by 2**(e-30) */
    PUT_BITS_DP64(((__INT8_T)(e - 30) + EXPBIAS_DP64) << EXPSHIFTBITS_DP64, r);
    z *= r;
  } else {
    /* Scale by 2**e */
    PUT_BITS_DP64(((__INT8_T)e + EXPBIAS_DP64) << EXPSHIFTBITS_DP64, r);
    z *= r;
  }

  return z;
}
#endif /* SQRT_AMD_INLINE */

#if defined(USE_SQRTF_AMD_INLINE)

static inline float
sqrtf_amd_inline(float x)
{
  /*
     Computes the square root of x.

     The calculation is carried out in three steps.

     Step 1. Reduction.
     The input argument is scaled to the interval [1, 4) by
     computing
               x = 2^e * y, where y in [1,4).
     Furthermore y is decomposed as y = c + t where
               c = 1 + j/32, j = 0,1,..,96; and |t| <= 1/64.

     Step 2. Approximation.
     An approximation q = sqrt(1 + (t/c)) - 1  is obtained
     from a basic series expansion using precomputed values
     stored in rt_jby32_lead_table_float and rt_jby32_trail_table_float.

     Step 3. Reconstruction.
     The value of sqrt(x) is reconstructed via
       sqrt(x) = 2^(e/2) * sqrt(y)
               = 2^(e/2) * sqrt(c) * sqrt(y/c)
               = 2^(e/2) * sqrt(c) * sqrt(1 + t/c)
               = 2^(e/2) * [ sqrt(c) + sqrt(c)*q ]
    */

  unsigned int ux, ax, u;
  float r1, r2, c, y, p, q, r, twop, z, rtc, rtc_lead, rtc_trail;
  int e, denorm = 0, index;

  /* Arrays rt_jby32_lead_table_float and rt_jby32_trail_table_float contain
     leading and trailing parts respectively of precomputed
     values of sqrt(j/32), for j = 32, 33, ..., 128.
     rt_jby32_lead_table_float contains the first 13 bits of precision,
     and rt_jby32_trail_table_float contains a further 24 bits precision. */

  static const float rt_jby32_lead_table_float[97] = {
      1.00000000000000000000e+00F,  /* 0x3f800000 */
      1.01538085937500000000e+00F,  /* 0x3f81f800 */
      1.03076171875000000000e+00F,  /* 0x3f83f000 */
      1.04565429687500000000e+00F,  /* 0x3f85d800 */
      1.06054687500000000000e+00F,  /* 0x3f87c000 */
      1.07519531250000000000e+00F,  /* 0x3f89a000 */
      1.08959960937500000000e+00F,  /* 0x3f8b7800 */
      1.10375976562500000000e+00F,  /* 0x3f8d4800 */
      1.11791992187500000000e+00F,  /* 0x3f8f1800 */
      1.13183593750000000000e+00F,  /* 0x3f90e000 */
      1.14550781250000000000e+00F,  /* 0x3f92a000 */
      1.15917968750000000000e+00F,  /* 0x3f946000 */
      1.17236328125000000000e+00F,  /* 0x3f961000 */
      1.18579101562500000000e+00F,  /* 0x3f97c800 */
      1.19873046875000000000e+00F,  /* 0x3f997000 */
      1.21191406250000000000e+00F,  /* 0x3f9b2000 */
      1.22460937500000000000e+00F,  /* 0x3f9cc000 */
      1.23730468750000000000e+00F,  /* 0x3f9e6000 */
      1.25000000000000000000e+00F,  /* 0x3fa00000 */
      1.26220703125000000000e+00F,  /* 0x3fa19000 */
      1.27465820312500000000e+00F,  /* 0x3fa32800 */
      1.28686523437500000000e+00F,  /* 0x3fa4b800 */
      1.29882812500000000000e+00F,  /* 0x3fa64000 */
      1.31079101562500000000e+00F,  /* 0x3fa7c800 */
      1.32275390625000000000e+00F,  /* 0x3fa95000 */
      1.33447265625000000000e+00F,  /* 0x3faad000 */
      1.34619140625000000000e+00F,  /* 0x3fac5000 */
      1.35766601562500000000e+00F,  /* 0x3fadc800 */
      1.36914062500000000000e+00F,  /* 0x3faf4000 */
      1.38061523437500000000e+00F,  /* 0x3fb0b800 */
      1.39184570312500000000e+00F,  /* 0x3fb22800 */
      1.40307617187500000000e+00F,  /* 0x3fb39800 */
      1.41406250000000000000e+00F,  /* 0x3fb50000 */
      1.42504882812500000000e+00F,  /* 0x3fb66800 */
      1.43603515625000000000e+00F,  /* 0x3fb7d000 */
      1.44677734375000000000e+00F,  /* 0x3fb93000 */
      1.45751953125000000000e+00F,  /* 0x3fba9000 */
      1.46826171875000000000e+00F,  /* 0x3fbbf000 */
      1.47900390625000000000e+00F,  /* 0x3fbd5000 */
      1.48950195312500000000e+00F,  /* 0x3fbea800 */
      1.50000000000000000000e+00F,  /* 0x3fc00000 */
      1.51025390625000000000e+00F,  /* 0x3fc15000 */
      1.52050781250000000000e+00F,  /* 0x3fc2a000 */
      1.53076171875000000000e+00F,  /* 0x3fc3f000 */
      1.54101562500000000000e+00F,  /* 0x3fc54000 */
      1.55102539062500000000e+00F,  /* 0x3fc68800 */
      1.56103515625000000000e+00F,  /* 0x3fc7d000 */
      1.57104492187500000000e+00F,  /* 0x3fc91800 */
      1.58105468750000000000e+00F,  /* 0x3fca6000 */
      1.59082031250000000000e+00F,  /* 0x3fcba000 */
      1.60058593750000000000e+00F,  /* 0x3fcce000 */
      1.61035156250000000000e+00F,  /* 0x3fce2000 */
      1.62011718750000000000e+00F,  /* 0x3fcf6000 */
      1.62963867187500000000e+00F,  /* 0x3fd09800 */
      1.63916015625000000000e+00F,  /* 0x3fd1d000 */
      1.64868164062500000000e+00F,  /* 0x3fd30800 */
      1.65820312500000000000e+00F,  /* 0x3fd44000 */
      1.66748046875000000000e+00F,  /* 0x3fd57000 */
      1.67700195312500000000e+00F,  /* 0x3fd6a800 */
      1.68627929687500000000e+00F,  /* 0x3fd7d800 */
      1.69555664062500000000e+00F,  /* 0x3fd90800 */
      1.70458984375000000000e+00F,  /* 0x3fda3000 */
      1.71386718750000000000e+00F,  /* 0x3fdb6000 */
      1.72290039062500000000e+00F,  /* 0x3fdc8800 */
      1.73193359375000000000e+00F,  /* 0x3fddb000 */
      1.74096679687500000000e+00F,  /* 0x3fded800 */
      1.75000000000000000000e+00F,  /* 0x3fe00000 */
      1.75878906250000000000e+00F,  /* 0x3fe12000 */
      1.76757812500000000000e+00F,  /* 0x3fe24000 */
      1.77636718750000000000e+00F,  /* 0x3fe36000 */
      1.78515625000000000000e+00F,  /* 0x3fe48000 */
      1.79394531250000000000e+00F,  /* 0x3fe5a000 */
      1.80273437500000000000e+00F,  /* 0x3fe6c000 */
      1.81127929687500000000e+00F,  /* 0x3fe7d800 */
      1.81982421875000000000e+00F,  /* 0x3fe8f000 */
      1.82836914062500000000e+00F,  /* 0x3fea0800 */
      1.83691406250000000000e+00F,  /* 0x3feb2000 */
      1.84545898437500000000e+00F,  /* 0x3fec3800 */
      1.85400390625000000000e+00F,  /* 0x3fed5000 */
      1.86230468750000000000e+00F,  /* 0x3fee6000 */
      1.87060546875000000000e+00F,  /* 0x3fef7000 */
      1.87915039062500000000e+00F,  /* 0x3ff08800 */
      1.88745117187500000000e+00F,  /* 0x3ff19800 */
      1.89550781250000000000e+00F,  /* 0x3ff2a000 */
      1.90380859375000000000e+00F,  /* 0x3ff3b000 */
      1.91210937500000000000e+00F,  /* 0x3ff4c000 */
      1.92016601562500000000e+00F,  /* 0x3ff5c800 */
      1.92822265625000000000e+00F,  /* 0x3ff6d000 */
      1.93627929687500000000e+00F,  /* 0x3ff7d800 */
      1.94433593750000000000e+00F,  /* 0x3ff8e000 */
      1.95239257812500000000e+00F,  /* 0x3ff9e800 */
      1.96044921875000000000e+00F,  /* 0x3ffaf000 */
      1.96826171875000000000e+00F,  /* 0x3ffbf000 */
      1.97631835937500000000e+00F,  /* 0x3ffcf800 */
      1.98413085937500000000e+00F,  /* 0x3ffdf800 */
      1.99194335937500000000e+00F,  /* 0x3ffef800 */
      2.00000000000000000000e+00F}; /* 0x40000000 */

  static const float rt_jby32_trail_table_float[97] = {
      0.00000000000000000000e+00F,  /* 0x00000000 */
      1.23941208585165441036e-04F,  /* 0x3901f637 */
      1.46876545841223560274e-05F,  /* 0x37766aff */
      1.70736297150142490864e-04F,  /* 0x393307ad */
      1.13296780909877270460e-04F,  /* 0x38ed99bf */
      9.53458802541717886925e-05F,  /* 0x38c7f46e */
      1.25126505736261606216e-04F,  /* 0x39033464 */
      2.10342666832730174065e-04F,  /* 0x395c8f6e */
      1.14066875539720058441e-04F,  /* 0x38ef3730 */
      8.72047676239162683487e-05F,  /* 0x38b6e1b4 */
      1.36111237225122749805e-04F,  /* 0x390eb915 */
      2.26244374061934649944e-05F,  /* 0x37bdc99c */
      2.40658700931817293167e-04F,  /* 0x397c5954 */
      6.31069415248930454254e-05F,  /* 0x38845848 */
      2.27412077947519719601e-04F,  /* 0x396e7577 */
      5.90185391047270968556e-06F,  /* 0x36c6088a */
      1.35496389702893793583e-04F,  /* 0x390e1409 */
      1.32179571664892137051e-04F,  /* 0x390a99af */
      0.00000000000000000000e+00F,  /* 0x00000000 */
      2.31086043640971183777e-04F,  /* 0x39724fb0 */
      9.66752704698592424393e-05F,  /* 0x38cabe24 */
      8.85332483449019491673e-05F,  /* 0x38b9aaed */
      2.09980673389509320259e-04F,  /* 0x395c2e42 */
      2.20044588786549866199e-04F,  /* 0x3966bbc5 */
      1.21749282698146998882e-04F,  /* 0x38ff53a6 */
      1.62125259521417319775e-04F,  /* 0x392a002b */
      9.97955357888713479042e-05F,  /* 0x38d14952 */
      1.81545779923908412457e-04F,  /* 0x393e5d53 */
      1.65768768056295812130e-04F,  /* 0x392dd237 */
      5.48927710042335093021e-05F,  /* 0x38663caa */
      9.53875860432162880898e-05F,  /* 0x38c80ad2 */
      4.53481625299900770187e-05F,  /* 0x383e3438 */
      1.51062369695864617825e-04F,  /* 0x391e667f */
      1.70453247847035527229e-04F,  /* 0x3932bbb2 */
      1.05505387182347476482e-04F,  /* 0x38dd42c6 */
      2.02269104192964732647e-04F,  /* 0x39541833 */
      2.18442466575652360916e-04F,  /* 0x39650db4 */
      1.55796806211583316326e-04F,  /* 0x39235d63 */
      1.60395247803535312414e-05F,  /* 0x37868c9e */
      4.49578510597348213196e-05F,  /* 0x383c9120 */
      0.00000000000000000000e+00F,  /* 0x00000000 */
      1.26840444863773882389e-04F,  /* 0x39050079 */
      1.82820076588541269302e-04F,  /* 0x393fb364 */
      1.69370483490638434887e-04F,  /* 0x3931990b */
      8.78757418831810355186e-05F,  /* 0x38b849ee */
      1.83815121999941766262e-04F,  /* 0x3940be7f */
      2.14343352126888930798e-04F,  /* 0x3960c15b */
      1.80714370799250900745e-04F,  /* 0x393d7e25 */
      8.41425862745381891727e-05F,  /* 0x38b075b5 */
      1.69945167726837098598e-04F,  /* 0x3932334f */
      1.95121858268976211548e-04F,  /* 0x394c99a0 */
      1.60778334247879683971e-04F,  /* 0x3928969b */
      6.79871009197086095810e-05F,  /* 0x388e944c */
      1.61929419846273958683e-04F,  /* 0x3929cb99 */
      1.99474830878898501396e-04F,  /* 0x39512a1e */
      1.81604162207804620266e-04F,  /* 0x393e6cff */
      1.09270178654696792364e-04F,  /* 0x38e527fb */
      2.27539261686615645885e-04F,  /* 0x396e979b */
      4.90300008095800876617e-05F,  /* 0x384da590 */
      6.28985289949923753738e-05F,  /* 0x3883e864 */
      2.58551553997676819563e-05F,  /* 0x37d8e386 */
      1.82868374395184218884e-04F,  /* 0x393fc05b */
      4.64625991298817098141e-05F,  /* 0x3842e0d6 */
      1.05703387816902250051e-04F,  /* 0x38ddad13 */
      1.17213814519345760345e-04F,  /* 0x38f5d0b0 */
      8.17377731436863541603e-05F,  /* 0x38ab6aa2 */
      0.00000000000000000000e+00F,  /* 0x00000000 */
      1.16847433673683553934e-04F,  /* 0x38f50bfd */
      1.88827965757809579372e-04F,  /* 0x3946001f */
      2.16612941585481166840e-04F,  /* 0x39632298 */
      2.00857131858356297016e-04F,  /* 0x39529d2d */
      1.42199307447299361229e-04F,  /* 0x39151b56 */
      4.12627305195201188326e-05F,  /* 0x382d1185 */
      1.42796401632949709892e-04F,  /* 0x3915bb9e */
      2.03253570361994206905e-04F,  /* 0x39552077 */
      2.23214170546270906925e-04F,  /* 0x396a0e99 */
      2.03244591830298304558e-04F,  /* 0x39551e0e */
      1.43898156238719820976e-04F,  /* 0x3916e35e */
      4.57155256299301981926e-05F,  /* 0x383fbeac */
      1.53365719597786664963e-04F,  /* 0x3920d0cc */
      2.23224633373320102692e-04F,  /* 0x396a1168 */
      1.16566716314991936088e-05F,  /* 0x37439106 */
      7.43694272387074306607e-06F,  /* 0x36f98ada */
      2.11048507480882108212e-04F,  /* 0x395d4ce7 */
      1.34682719362899661064e-04F,  /* 0x390d399e */
      2.29425968427676707506e-05F,  /* 0x37c074da */
      1.20421340398024767637e-04F,  /* 0x38fc8ab7 */
      1.83421318070031702518e-04F,  /* 0x394054c9 */
      2.12376224226318299770e-04F,  /* 0x395eb14f */
      2.07710763788782060146e-04F,  /* 0x3959ccef */
      1.69840845046564936638e-04F,  /* 0x3932174e */
      9.91739216260612010956e-05F,  /* 0x38cffb98 */
      2.40249748458154499531e-04F,  /* 0x397beb8d */
      1.05178231024183332920e-04F,  /* 0x38dc9322 */
      1.82623916771262884140e-04F,  /* 0x393f7ebc */
      2.28821940254420042038e-04F,  /* 0x396fefec */
      0.00000000000000000000e+00F}; /* 0x00000000 */

  /* Handle special arguments first */

  GET_BITS_SP32(x, ux);
  ax = ux & (~SIGNBIT_SP32);

  if (ax >= 0x7f800000) {
    /* x is either NaN or infinity */
    if (ux & MANTBITS_SP32)
      /* x is NaN */
      return x + x; /* Raise invalid if it is a signalling NaN */
    else if (ux & SIGNBIT_SP32)
      return nanf_with_flags(AMD_F_INVALID);
    else
      /* x is positive infinity */
      return x;
  } else if (ux & SIGNBIT_SP32) {
    /* x is negative. */
    if (x == 0.0F)
      /* Handle negative zero first */
      return x;
    else
      return nanf_with_flags(AMD_F_INVALID);
  } else if (ux <= 0x007fffff) {
    /* x is denormalised or zero */
    if (ux == 0)
      /* x is zero */
      return x;
    else {
      /* x is denormalised; scale it up */
      /* Normalize x by increasing the exponent by 26
         and subtracting a correction to account for the implicit
         bit. This replaces a slow denormalized
         multiplication by a fast normal subtraction. */
      static const float corr = 7.888609052210118054e-31F; /* 0x0d800000 */
      denorm = 1;
      GET_BITS_SP32(x, ux);
      PUT_BITS_SP32(ux | 0x0d800000, x);
      x -= corr;
      GET_BITS_SP32(x, ux);
    }
  }

  /* Main algorithm */

  /*
     Find y and e such that x = 2^e * y, where y in [1,4).
     This is done using an in-lined variant of splitFloat,
     which also ensures that e is even.
   */
  y = x;
  ux &= EXPBITS_SP32;
  ux >>= EXPSHIFTBITS_SP32;
  if (ux & 1) {
    GET_BITS_SP32(y, u);
    u &= (SIGNBIT_SP32 | MANTBITS_SP32);
    u |= ONEEXPBITS_SP32;
    PUT_BITS_SP32(u, y);
    e = ux - EXPBIAS_SP32;
  } else {
    GET_BITS_SP32(y, u);
    u &= (SIGNBIT_SP32 | MANTBITS_SP32);
    u |= TWOEXPBITS_SP32;
    PUT_BITS_SP32(u, y);
    e = ux - EXPBIAS_SP32 - 1;
  }

  /* Find the index of the sub-interval of [1,4) in which y lies. */

  index = (int)(32.0F * y + 0.5);

  /* Look up the table values and compute c and r = c/t */

  rtc_lead = rt_jby32_lead_table_float[index - 32];
  rtc_trail = rt_jby32_trail_table_float[index - 32];
  c = 0.03125F * index;
  r = (y - c) / c;

  /*
  Find q = sqrt(1+r) - 1.
  From one step of Newton on (q+1)^2 = 1+r
  */

  p = r * 0.5F - r * r * (0.1250079870F - r * (0.6250522999e-01F));
  twop = p + p;
  q = p - (p * p + (twop - r)) / (twop + 2.0);

  /* Reconstruction */

  rtc = rtc_lead + rtc_trail;
  e >>= 1; /* e = e/2 */
  z = rtc_lead + (rtc * q + rtc_trail);

  if (denorm) {
    /* Scale by 2**(e-13) */
    PUT_BITS_SP32(((e - 13) + EXPBIAS_SP32) << EXPSHIFTBITS_SP32, r);
    z *= r;
  } else {
    /* Scale by 2**e */
    PUT_BITS_SP32((e + EXPBIAS_SP32) << EXPSHIFTBITS_SP32, r);
    z *= r;
  }

  return z;
}
#endif /* SQRTF_AMD_INLINE */

#ifdef USE_LOG_KERNEL_AMD
static inline void
log_kernel_amd64(double x, __UINT8_T ux, int *xexp, double *r1, double *r2)
{

  int expadjust;
  double r, z1, z2, correction, f, f1, f2, q, u, v, poly;
  int index;

  /*
    Computes natural log(x). Algorithm based on:
    Ping-Tak Peter Tang
    "Table-driven implementation of the logarithm function in IEEE
    floating-point arithmetic"
    ACM Transactions on Mathematical Software (TOMS)
    Volume 16, Issue 4 (December 1990)
  */

  /* Arrays ln_lead_table and ln_tail_table contain
     leading and trailing parts respectively of precomputed
     values of natural log(1+i/64), for i = 0, 1, ..., 64.
     ln_lead_table contains the first 24 bits of precision,
     and ln_tail_table contains a further 53 bits precision. */

  static const double ln_lead_table[65] = {
      0.00000000000000000000e+00,  /* 0x0000000000000000 */
      1.55041813850402832031e-02,  /* 0x3f8fc0a800000000 */
      3.07716131210327148438e-02,  /* 0x3f9f829800000000 */
      4.58095073699951171875e-02,  /* 0x3fa7745800000000 */
      6.06245994567871093750e-02,  /* 0x3faf0a3000000000 */
      7.52233862876892089844e-02,  /* 0x3fb341d700000000 */
      8.96121263504028320312e-02,  /* 0x3fb6f0d200000000 */
      1.03796780109405517578e-01,  /* 0x3fba926d00000000 */
      1.17783010005950927734e-01,  /* 0x3fbe270700000000 */
      1.31576299667358398438e-01,  /* 0x3fc0d77e00000000 */
      1.45181953907012939453e-01,  /* 0x3fc2955280000000 */
      1.58604979515075683594e-01,  /* 0x3fc44d2b00000000 */
      1.71850204467773437500e-01,  /* 0x3fc5ff3000000000 */
      1.84922337532043457031e-01,  /* 0x3fc7ab8900000000 */
      1.97825729846954345703e-01,  /* 0x3fc9525a80000000 */
      2.10564732551574707031e-01,  /* 0x3fcaf3c900000000 */
      2.23143517971038818359e-01,  /* 0x3fcc8ff780000000 */
      2.35566020011901855469e-01,  /* 0x3fce270700000000 */
      2.47836112976074218750e-01,  /* 0x3fcfb91800000000 */
      2.59957492351531982422e-01,  /* 0x3fd0a324c0000000 */
      2.71933674812316894531e-01,  /* 0x3fd1675c80000000 */
      2.83768117427825927734e-01,  /* 0x3fd22941c0000000 */
      2.95464158058166503906e-01,  /* 0x3fd2e8e280000000 */
      3.07025015354156494141e-01,  /* 0x3fd3a64c40000000 */
      3.18453729152679443359e-01,  /* 0x3fd4618bc0000000 */
      3.29753279685974121094e-01,  /* 0x3fd51aad80000000 */
      3.40926527976989746094e-01,  /* 0x3fd5d1bd80000000 */
      3.51976394653320312500e-01,  /* 0x3fd686c800000000 */
      3.62905442714691162109e-01,  /* 0x3fd739d7c0000000 */
      3.73716354370117187500e-01,  /* 0x3fd7eaf800000000 */
      3.84411692619323730469e-01,  /* 0x3fd89a3380000000 */
      3.94993782043457031250e-01,  /* 0x3fd9479400000000 */
      4.05465066432952880859e-01,  /* 0x3fd9f323c0000000 */
      4.15827870368957519531e-01,  /* 0x3fda9cec80000000 */
      4.26084339618682861328e-01,  /* 0x3fdb44f740000000 */
      4.36236739158630371094e-01,  /* 0x3fdbeb4d80000000 */
      4.46287095546722412109e-01,  /* 0x3fdc8ff7c0000000 */
      4.56237375736236572266e-01,  /* 0x3fdd32fe40000000 */
      4.66089725494384765625e-01,  /* 0x3fddd46a00000000 */
      4.75845873355865478516e-01,  /* 0x3fde744240000000 */
      4.85507786273956298828e-01,  /* 0x3fdf128f40000000 */
      4.95077252388000488281e-01,  /* 0x3fdfaf5880000000 */
      5.04556000232696533203e-01,  /* 0x3fe02552a0000000 */
      5.13945698738098144531e-01,  /* 0x3fe0723e40000000 */
      5.23248136043548583984e-01,  /* 0x3fe0be72e0000000 */
      5.32464742660522460938e-01,  /* 0x3fe109f380000000 */
      5.41597247123718261719e-01,  /* 0x3fe154c3c0000000 */
      5.50647079944610595703e-01,  /* 0x3fe19ee6a0000000 */
      5.59615731239318847656e-01,  /* 0x3fe1e85f40000000 */
      5.68504691123962402344e-01,  /* 0x3fe23130c0000000 */
      5.77315330505371093750e-01,  /* 0x3fe2795e00000000 */
      5.86049020290374755859e-01,  /* 0x3fe2c0e9e0000000 */
      5.94707071781158447266e-01,  /* 0x3fe307d720000000 */
      6.03290796279907226562e-01,  /* 0x3fe34e2880000000 */
      6.11801505088806152344e-01,  /* 0x3fe393e0c0000000 */
      6.20240390300750732422e-01,  /* 0x3fe3d90260000000 */
      6.28608644008636474609e-01,  /* 0x3fe41d8fe0000000 */
      6.36907458305358886719e-01,  /* 0x3fe4618bc0000000 */
      6.45137906074523925781e-01,  /* 0x3fe4a4f840000000 */
      6.53301239013671875000e-01,  /* 0x3fe4e7d800000000 */
      6.61398470401763916016e-01,  /* 0x3fe52a2d20000000 */
      6.69430613517761230469e-01,  /* 0x3fe56bf9c0000000 */
      6.77398800849914550781e-01,  /* 0x3fe5ad4040000000 */
      6.85303986072540283203e-01,  /* 0x3fe5ee02a0000000 */
      6.93147122859954833984e-01}; /* 0x3fe62e42e0000000 */

  static const double ln_tail_table[65] = {
      0.00000000000000000000e+00,  /* 0x0000000000000000 */
      5.15092497094772879206e-09,  /* 0x3e361f807c79f3db */
      4.55457209735272790188e-08,  /* 0x3e6873c1980267c8 */
      2.86612990859791781788e-08,  /* 0x3e5ec65b9f88c69e */
      2.23596477332056055352e-08,  /* 0x3e58022c54cc2f99 */
      3.49498983167142274770e-08,  /* 0x3e62c37a3a125330 */
      3.23392843005887000414e-08,  /* 0x3e615cad69737c93 */
      1.35722380472479366661e-08,  /* 0x3e4d256ab1b285e9 */
      2.56504325268044191098e-08,  /* 0x3e5b8abcb97a7aa2 */
      5.81213608741512136843e-08,  /* 0x3e6f34239659a5dc */
      5.59374849578288093334e-08,  /* 0x3e6e07fd48d30177 */
      5.06615629004996189970e-08,  /* 0x3e6b32df4799f4f6 */
      5.24588857848400955725e-08,  /* 0x3e6c29e4f4f21cf8 */
      9.61968535632653505972e-10,  /* 0x3e1086c848df1b59 */
      1.34829655346594463137e-08,  /* 0x3e4cf456b4764130 */
      3.65557749306383026498e-08,  /* 0x3e63a02ffcb63398 */
      3.33431709374069198903e-08,  /* 0x3e61e6a6886b0976 */
      5.13008650536088382197e-08,  /* 0x3e6b8abcb97a7aa2 */
      5.09285070380306053751e-08,  /* 0x3e6b578f8aa35552 */
      3.20853940845502057341e-08,  /* 0x3e6139c871afb9fc */
      4.06713248643004200446e-08,  /* 0x3e65d5d30701ce64 */
      5.57028186706125221168e-08,  /* 0x3e6de7bcb2d12142 */
      5.48356693724804282546e-08,  /* 0x3e6d708e984e1664 */
      1.99407553679345001938e-08,  /* 0x3e556945e9c72f36 */
      1.96585517245087232086e-09,  /* 0x3e20e2f613e85bda */
      6.68649386072067321503e-09,  /* 0x3e3cb7e0b42724f6 */
      5.89936034642113390002e-08,  /* 0x3e6fac04e52846c7 */
      2.85038578721554472484e-08,  /* 0x3e5e9b14aec442be */
      5.09746772910284482606e-08,  /* 0x3e6b5de8034e7126 */
      5.54234668933210171467e-08,  /* 0x3e6dc157e1b259d3 */
      6.29100830926604004874e-09,  /* 0x3e3b05096ad69c62 */
      2.61974119468563937716e-08,  /* 0x3e5c2116faba4cdd */
      4.16752115011186398935e-08,  /* 0x3e665fcc25f95b47 */
      2.47747534460820790327e-08,  /* 0x3e5a9a08498d4850 */
      5.56922172017964209793e-08,  /* 0x3e6de647b1465f77 */
      2.76162876992552906035e-08,  /* 0x3e5da71b7bf7861d */
      7.08169709942321478061e-09,  /* 0x3e3e6a6886b09760 */
      5.77453510221151779025e-08,  /* 0x3e6f0075eab0ef64 */
      4.43021445893361960146e-09,  /* 0x3e33071282fb989b */
      3.15140984357495864573e-08,  /* 0x3e60eb43c3f1bed2 */
      2.95077445089736670973e-08,  /* 0x3e5faf06ecb35c84 */
      1.44098510263167149349e-08,  /* 0x3e4ef1e63db35f68 */
      1.05196987538551827693e-08,  /* 0x3e469743fb1a71a5 */
      5.23641361722697546261e-08,  /* 0x3e6c1cdf404e5796 */
      7.72099925253243069458e-09,  /* 0x3e4094aa0ada625e */
      5.62089493829364197156e-08,  /* 0x3e6e2d4c96fde3ec */
      3.53090261098577946927e-08,  /* 0x3e62f4d5e9a98f34 */
      3.80080516835568242269e-08,  /* 0x3e6467c96ecc5cbe */
      5.66961038386146408282e-08,  /* 0x3e6e7040d03dec5a */
      4.42287063097349852717e-08,  /* 0x3e67bebf4282de36 */
      3.45294525105681104660e-08,  /* 0x3e6289b11aeb783f */
      2.47132034530447431509e-08,  /* 0x3e5a891d1772f538 */
      3.59655343422487209774e-08,  /* 0x3e634f10be1fb591 */
      5.51581770357780862071e-08,  /* 0x3e6d9ce1d316eb93 */
      3.60171867511861372793e-08,  /* 0x3e63562a19a9c442 */
      1.94511067964296180547e-08,  /* 0x3e54e2adf548084c */
      1.54137376631349347838e-08,  /* 0x3e508ce55cc8c97a */
      3.93171034490174464173e-09,  /* 0x3e30e2f613e85bda */
      5.52990607758839766440e-08,  /* 0x3e6db03ebb0227bf */
      3.29990737637586136511e-08,  /* 0x3e61b75bb09cb098 */
      1.18436010922446096216e-08,  /* 0x3e496f16abb9df22 */
      4.04248680368301346709e-08,  /* 0x3e65b3f399411c62 */
      2.27418915900284316293e-08,  /* 0x3e586b3e59f65355 */
      1.70263791333409206020e-08,  /* 0x3e52482ceae1ac12 */
      5.76999904754328540596e-08}; /* 0x3e6efa39ef35793c */

  /* Approximating polynomial coefficients for x near 1.0 */
  static const double ca_1 =
                          8.33333333333317923934e-02, /* 0x3fb55555555554e6 */
      ca_2 = 1.25000000037717509602e-02,              /* 0x3f89999999bac6d4 */
      ca_3 = 2.23213998791944806202e-03,              /* 0x3f62492307f1519f */
      ca_4 = 4.34887777707614552256e-04;              /* 0x3f3c8034c85dfff0 */

  /* Approximating polynomial coefficients for other x */
  static const double cb_1 =
                          8.33333333333333593622e-02, /* 0x3fb5555555555557 */
      cb_2 = 1.24999999978138668903e-02,              /* 0x3f89999999865ede */
      cb_3 = 2.23219810758559851206e-03;              /* 0x3f6249423bd94741 */

  static const __UINT8_T log_thresh1 = 0x3fee0faa00000000,
                         log_thresh2 = 0x3ff1082c00000000;

  /* log_thresh1 = 9.39412117004394531250e-1 = 0x3fee0faa00000000
     log_thresh2 = 1.06449508666992187500 = 0x3ff1082c00000000 */
  if (ux >= log_thresh1 && ux <= log_thresh2) {
    /* Arguments close to 1.0 are handled separately to maintain
       accuracy.

       The approximation in this region exploits the identity
           log( 1 + r ) = log( 1 + u/2 )  /  log( 1 - u/2 ), where
           u  = 2r / (2+r).
       Note that the right hand side has an odd Taylor series expansion
       which converges much faster than the Taylor series expansion of
       log( 1 + r ) in r. Thus, we approximate log( 1 + r ) by
           u + A1 * u^3 + A2 * u^5 + ... + An * u^(2n+1).

       One subtlety is that since u cannot be calculated from
       r exactly, the rounding error in the first u should be
       avoided if possible. To accomplish this, we observe that
                     u  =  r  -  r*r/(2+r).
       Since x (=1+r) is the input argument, and thus presumed exact,
       the formula above approximates u accurately because
                     u  =  r  -  correction,
       and the magnitude of "correction" (of the order of r*r)
       is small.
       With these observations, we will approximate log( 1 + r ) by
          r + (  (A1*u^3 + ... + An*u^(2n+1)) - correction ).

       We approximate log(1+r) by an odd polynomial in u, where
                u = 2r/(2+r) = r - r*r/(2+r).
    */
    r = x - 1.0;
    u = r / (2.0 + r);
    correction = r * u;
    u = u + u;
    v = u * u;
    z1 = r;
    z2 = (u * v * (ca_1 + v * (ca_2 + v * (ca_3 + v * ca_4))) - correction);
    *r1 = z1;
    *r2 = z2;
    *xexp = 0;
  } else {
    /*
      First, we decompose the argument x to the form
      x  =  2**M  *  (F1  +  F2),
      where  1 <= F1+F2 < 2, M has the value of an integer,
      F1 = 1 + j/64, j ranges from 0 to 64, and |F2| <= 1/128.

      Second, we approximate log( 1 + F2/F1 ) by an odd polynomial
      in U, where U  =  2 F2 / (2 F2 + F1).
      Note that log( 1 + F2/F1 ) = log( 1 + U/2 ) - log( 1 - U/2 ).
      The core approximation calculates
      Poly = [log( 1 + U/2 ) - log( 1 - U/2 )]/U   -   1.
      Note that  log(1 + U/2) - log(1 - U/2) = 2 arctanh ( U/2 ),
      thus, Poly =  2 arctanh( U/2 ) / U  -  1.

      It is not hard to see that
        log(x) = M*log(2) + log(F1) + log( 1 + F2/F1 ).
      Hence, we return Z1 = log(F1), and  Z2 = log( 1 + F2/F1).
      The values of log(F1) are calculated beforehand and stored
      in the program.
    */

    f = x;
    if (ux < IMPBIT_DP64) {
      /* The input argument x is denormalized */
      /* Normalize f by increasing the exponent by 60
         and subtracting a correction to account for the implicit
         bit. This replaces a slow denormalized
         multiplication by a fast normal subtraction. */
      static const double corr =
          2.5653355008114851558350183e-290; /* 0x03d0000000000000 */
      GET_BITS_DP64(f, ux);
      ux |= 0x03d0000000000000;
      PUT_BITS_DP64(ux, f);
      f -= corr;
      GET_BITS_DP64(f, ux);
      expadjust = 60;
    } else
      expadjust = 0;

    /* Store the exponent of x in xexp and put
       f into the range [0.5,1) */
    *xexp = (int)((ux & EXPBITS_DP64) >> EXPSHIFTBITS_DP64) - EXPBIAS_DP64 -
            expadjust;
    PUT_BITS_DP64((ux & MANTBITS_DP64) | HALFEXPBITS_DP64, f);

    /* Now  x = 2**xexp  * f,  1/2 <= f < 1. */

    /* Set index to be the nearest integer to 128*f */
    r = 128.0 * f;
    index = (int)(r + 0.5);

    z1 = ln_lead_table[index - 64];
    q = ln_tail_table[index - 64];
    f1 = index * 0.0078125; /* 0.0078125 = 1/128 */
    f2 = f - f1;
    /* At this point, x = 2**xexp * ( f1  +  f2 ) where
       f1 = j/128, j = 64, 65, ..., 128 and |f2| <= 1/256. */

    /* Calculate u = 2 f2 / ( 2 f1 + f2 ) = f2 / ( f1 + 0.5*f2 ) */
    /* u = f2 / (f1 + 0.5 * f2); */
    u = f2 / (f1 + 0.5 * f2);

    /* Here, |u| <= 2(exp(1/16)-1) / (exp(1/16)+1).
       The core approximation calculates
       poly = [log(1 + u/2) - log(1 - u/2)]/u  -  1  */
    v = u * u;
    poly = (v * (cb_1 + v * (cb_2 + v * cb_3)));
    z2 = q + (u + u * poly);
    *r1 = z1;
    *r2 = z2;
  }
  return;
}
#endif /* USE_LOG_KERNEL_AMD */

#if defined(USE_REMAINDER_PIBY2F_INLINE)
/* Define this to get debugging print statements activated */
#define DEBUGGING_PRINT
#undef DEBUGGING_PRINT

#ifdef DEBUGGING_PRINT
#include <stdio.h>
char *
d2b(__INT8_T d, int bitsper, int point)
{
  static char buff[200];
  int i, j;
  j = bitsper;
  if (point >= 0 && point <= bitsper)
    j++;
  buff[j] = '\0';
  for (i = bitsper - 1; i >= 0; i--) {
    j--;
    if (d % 2 == 1)
      buff[j] = '1';
    else
      buff[j] = '0';
    if (i == point) {
      j--;
      buff[j] = '.';
    }
    d /= 2;
  }
  return buff;
}
#endif

/* Given positive argument x, reduce it to the range [-pi/4,pi/4] using
   extra precision, and return the result in r.
   Return value "region" tells how many lots of pi/2 were subtracted
   from x to put it in the range [-pi/4,pi/4], mod 4. */
static inline void
__remainder_piby2f_inline(__UINT8_T ux, double *r, int *region)
{

/* This method simulates multi-precision floating-point
   arithmetic and is accurate for all 1 <= x < infinity */
#define bitsper 36
  __UINT8_T res[10] = { 0 };
  __UINT8_T u, carry, mask, mant, nextbits;
  int first, last, i, rexp, xexp, resexp, ltb, determ, bc;
  double dx;
  static const double piby2 =
      1.57079632679489655800e+00; /* 0x3ff921fb54442d18 */
  static __UINT8_T pibits[] = {
      0L,           5215L,        13000023176L, 11362338026L, 67174558139L,
      34819822259L, 10612056195L, 67816420731L, 57840157550L, 19558516809L,
      50025467026L, 25186875954L, 18152700886L};

#ifdef DEBUGGING_PRINT
  printf("On entry, x = %25.20e = %s\n", x, double2hex(&x));
#endif

  xexp = (int)(((ux & EXPBITS_DP64) >> EXPSHIFTBITS_DP64) - EXPBIAS_DP64);
  ux = ((ux & MANTBITS_DP64) | IMPBIT_DP64) >> 29;

#ifdef DEBUGGING_PRINT
  printf("ux = %s\n", d2b(ux, 64, -1));
#endif

  /* Now ux is the mantissa bit pattern of x as a __INT8_T integer */
  mask = 1;
  mask = (mask << bitsper) - 1;

  /* Set first and last to the positions of the first
     and last chunks of 2/pi that we need */
  first = xexp / bitsper;
  resexp = xexp - first * bitsper;
  /* 120 is the theoretical maximum number of bits (actually
     115 for IEEE single precision) that we need to extract
     from the middle of 2/pi to compute the reduced argument
     accurately enough for our purposes */
  last = first + 120 / bitsper;

#ifdef DEBUGGING_PRINT
  printf("first = %d, last = %d\n", first, last);
#endif

/* Do a __INT8_T multiplication of the bits of 2/pi by the
   integer mantissa */
  /* Unroll the loop. This is only correct because we know
     that bitsper is fixed as 36. */
  res[4] = 0;
  u = pibits[last] * ux;
  res[3] = u & mask;
  carry = u >> bitsper;
  u = pibits[last - 1] * ux + carry;
  res[2] = u & mask;
  carry = u >> bitsper;
  u = pibits[last - 2] * ux + carry;
  res[1] = u & mask;
  carry = u >> bitsper;
  u = pibits[first] * ux + carry;
  res[0] = u & mask;

#ifdef DEBUGGING_PRINT
  printf("resexp = %d\n", resexp);
  printf("Significant part of x * 2/pi with binary"
         " point in correct place:\n");
  for (i = 0; i <= last - first; i++) {
    if (i > 0 && i % 5 == 0)
      printf("\n ");
    if (i == 1)
      printf("%s ", d2b(res[i], bitsper, resexp));
    else
      printf("%s ", d2b(res[i], bitsper, -1));
  }
  printf("\n");
#endif

  /* Reconstruct the result */
  ltb = (int)((((res[0] << bitsper) | res[1]) >> (bitsper - 1 - resexp)) & 7);

  /* determ says whether the fractional part is >= 0.5 */
  determ = ltb & 1;

#ifdef DEBUGGING_PRINT
  printf("ltb = %d (last two bits before binary point"
         " and first bit after)\n",
         ltb);
  printf("determ = %d (1 means need to negate because the fractional\n"
         "            part of x * 2/pi is greater than 0.5)\n",
         determ);
#endif

  i = 1;
  if (determ) {
    /* The mantissa is >= 0.5. We want to subtract it
       from 1.0 by negating all the bits */
    *region = ((ltb >> 1) + 1) & 3;
    mant = 1;
    mant = ~(res[1]) & ((mant << (bitsper - resexp)) - 1);
    while (mant < 0x0000000000010000) {
      i++;
      mant = (mant << bitsper) | (~(res[i]) & mask);
    }
    nextbits = (~(res[i + 1]) & mask);
  } else {
    *region = (ltb >> 1);
    mant = 1;
    mant = res[1] & ((mant << (bitsper - resexp)) - 1);
    while (mant < 0x0000000000010000) {
      i++;
      mant = (mant << bitsper) | res[i];
    }
    nextbits = res[i + 1];
  }

#ifdef DEBUGGING_PRINT
  printf("First bits of mant = %s\n", d2b(mant, bitsper, -1));
#endif

  /* Normalize the mantissa. The shift value 6 here, determined by
     trial and error, seems to give optimal speed. */
  bc = 0;
  while (mant < 0x0000400000000000) {
    bc += 6;
    mant <<= 6;
  }
  while (mant < 0x0010000000000000) {
    bc++;
    mant <<= 1;
  }
  mant |= nextbits >> (bitsper - bc);

  rexp = 52 + resexp - bc - i * bitsper;

#ifdef DEBUGGING_PRINT
  printf("Normalised mantissa = 0x%016lx\n", mant);
  printf("Exponent to be inserted on mantissa = rexp = %d\n", rexp);
#endif

  /* Put the result exponent rexp onto the mantissa pattern */
  u = ((__UINT8_T)rexp + EXPBIAS_DP64) << EXPSHIFTBITS_DP64;
  ux = (mant & MANTBITS_DP64) | u;
  if (determ)
    /* If we negated the mantissa we negate x too */
    ux |= SIGNBIT_DP64;
  PUT_BITS_DP64(ux, dx);

#ifdef DEBUGGING_PRINT
  printf("(x*2/pi) = %25.20e = %s\n", dx, double2hex(&dx));
#endif

  /* x is a double precision version of the fractional part of
     x * 2 / pi. Multiply x by pi/2 in double precision
     to get the reduced argument r. */
  *r = dx * piby2;

#ifdef DEBUGGING_PRINT
  printf(" r = frac(x*2/pi) * pi/2:\n");
  printf(" r = %25.20e = %s\n", *r, double2hex(r));
  printf("region = (number of pi/2 subtracted from x) mod 4 = %d\n", *region);
#endif
}
#endif /* USE_REMAINDER_PIBY2F_INLINE */

#if defined(USE_REMAINDER_PIBY2_INLINE)
/* Given positive argument x, reduce it to the range [-pi/4,pi/4] using
   extra precision, and return the result in r, rr.
   Return value "region" tells how many lots of pi/2 were subtracted
   from x to put it in the range [-pi/4,pi/4], mod 4. */
static inline void
__remainder_piby2_inline(double x, double *r, double *rr, int *region)
{

  /* This method simulates multi-precision floating-point
     arithmetic and is accurate for all 1 <= x < infinity */
  static const double piby2_lead =
                          1.57079632679489655800e+00, /* 0x3ff921fb54442d18 */
      piby2_part1 = 1.57079631090164184570e+00,       /* 0x3ff921fb50000000 */
      piby2_part2 = 1.58932547122958567343e-08,       /* 0x3e5110b460000000 */
      piby2_part3 = 6.12323399573676480327e-17;       /* 0x3c91a62633145c06 */
  const int bitsper = 10;
  __UINT8_T res[500];
  __UINT8_T ux, u, carry, mask, mant, highbitsrr;
  int first, last, i, rexp, xexp, resexp, ltb, determ;
  double xx, t;
  static __UINT8_T pibits[] = {
      0,   0,   0,   0,   0,   0,    162, 998,  54,  915,  580, 84,  671, 777,
      855, 839, 851, 311, 448, 877,  553, 358,  316, 270,  260, 127, 593, 398,
      701, 942, 965, 390, 882, 283,  570, 265,  221, 184,  6,   292, 750, 642,
      465, 584, 463, 903, 491, 114,  786, 617,  830, 930,  35,  381, 302, 749,
      72,  314, 412, 448, 619, 279,  894, 260,  921, 117,  569, 525, 307, 637,
      156, 529, 504, 751, 505, 160,  945, 1022, 151, 1023, 480, 358, 15,  956,
      753, 98,  858, 41,  721, 987,  310, 507,  242, 498,  777, 733, 244, 399,
      870, 633, 510, 651, 373, 158,  940, 506,  997, 965,  947, 833, 825, 990,
      165, 164, 746, 431, 949, 1004, 287, 565,  464, 533,  515, 193, 111, 798};

  GET_BITS_DP64(x, ux);

#ifdef DEBUGGING_PRINT
  printf("On entry, x = %25.20e = %s\n", x, double2hex(&x));
#endif

  xexp = (int)(((ux & EXPBITS_DP64) >> EXPSHIFTBITS_DP64) - EXPBIAS_DP64);
  ux = (ux & MANTBITS_DP64) | IMPBIT_DP64;

  /* Now ux is the mantissa bit pattern of x as a __INT8_T integer */
  carry = 0;
  mask = 1;
  mask = (mask << bitsper) - 1;

  /* Set first and last to the positions of the first
     and last chunks of 2/pi that we need */
  first = xexp / bitsper;
  resexp = xexp - first * bitsper;
  /* 180 is the theoretical maximum number of bits (actually
     175 for IEEE double precision) that we need to extract
     from the middle of 2/pi to compute the reduced argument
     accurately enough for our purposes */
  last = first + 180 / bitsper;

/* Do a __INT8_T multiplication of the bits of 2/pi by the
   integer mantissa */
  /* Unroll the loop. This is only correct because we know
     that bitsper is fixed as 10. */
  res[19] = 0;
  u = pibits[last] * ux;
  res[18] = u & mask;
  carry = u >> bitsper;
  u = pibits[last - 1] * ux + carry;
  res[17] = u & mask;
  carry = u >> bitsper;
  u = pibits[last - 2] * ux + carry;
  res[16] = u & mask;
  carry = u >> bitsper;
  u = pibits[last - 3] * ux + carry;
  res[15] = u & mask;
  carry = u >> bitsper;
  u = pibits[last - 4] * ux + carry;
  res[14] = u & mask;
  carry = u >> bitsper;
  u = pibits[last - 5] * ux + carry;
  res[13] = u & mask;
  carry = u >> bitsper;
  u = pibits[last - 6] * ux + carry;
  res[12] = u & mask;
  carry = u >> bitsper;
  u = pibits[last - 7] * ux + carry;
  res[11] = u & mask;
  carry = u >> bitsper;
  u = pibits[last - 8] * ux + carry;
  res[10] = u & mask;
  carry = u >> bitsper;
  u = pibits[last - 9] * ux + carry;
  res[9] = u & mask;
  carry = u >> bitsper;
  u = pibits[last - 10] * ux + carry;
  res[8] = u & mask;
  carry = u >> bitsper;
  u = pibits[last - 11] * ux + carry;
  res[7] = u & mask;
  carry = u >> bitsper;
  u = pibits[last - 12] * ux + carry;
  res[6] = u & mask;
  carry = u >> bitsper;
  u = pibits[last - 13] * ux + carry;
  res[5] = u & mask;
  carry = u >> bitsper;
  u = pibits[last - 14] * ux + carry;
  res[4] = u & mask;
  carry = u >> bitsper;
  u = pibits[last - 15] * ux + carry;
  res[3] = u & mask;
  carry = u >> bitsper;
  u = pibits[last - 16] * ux + carry;
  res[2] = u & mask;
  carry = u >> bitsper;
  u = pibits[last - 17] * ux + carry;
  res[1] = u & mask;
  carry = u >> bitsper;
  u = pibits[last - 18] * ux + carry;
  res[0] = u & mask;

#ifdef DEBUGGING_PRINT
  printf("resexp = %d\n", resexp);
  printf("Significant part of x * 2/pi with binary"
         " point in correct place:\n");
  for (i = 0; i <= last - first; i++) {
    if (i > 0 && i % 5 == 0)
      printf("\n ");
    if (i == 1)
      printf("%s ", d2b((int)res[i], bitsper, resexp));
    else
      printf("%s ", d2b((int)res[i], bitsper, -1));
  }
  printf("\n");
#endif

  /* Reconstruct the result */
  ltb = (int)((((res[0] << bitsper) | res[1]) >> (bitsper - 1 - resexp)) & 7);

  /* determ says whether the fractional part is >= 0.5 */
  determ = ltb & 1;

#ifdef DEBUGGING_PRINT
  printf("ltb = %d (last two bits before binary point"
         " and first bit after)\n",
         ltb);
  printf("determ = %d (1 means need to negate because the fractional\n"
         "            part of x * 2/pi is greater than 0.5)\n",
         determ);
#endif

  i = 1;
  if (determ) {
    /* The mantissa is >= 0.5. We want to subtract it
       from 1.0 by negating all the bits */
    *region = ((ltb >> 1) + 1) & 3;
    mant = 1;
    mant = ~(res[1]) & ((mant << (bitsper - resexp)) - 1);
    while (mant < 0x0020000000000000) {
      i++;
      mant = (mant << bitsper) | (~(res[i]) & mask);
    }
    highbitsrr = ~(res[i + 1]) << (64 - bitsper);
  } else {
    *region = (ltb >> 1);
    mant = 1;
    mant = res[1] & ((mant << (bitsper - resexp)) - 1);
    while (mant < 0x0020000000000000) {
      i++;
      mant = (mant << bitsper) | res[i];
    }
    highbitsrr = res[i + 1] << (64 - bitsper);
  }

  rexp = 52 + resexp - i * bitsper;

  while (mant >= 0x0020000000000000) {
    rexp++;
    highbitsrr = (highbitsrr >> 1) | ((mant & 1) << 63);
    mant >>= 1;
  }

#ifdef DEBUGGING_PRINT
  printf("Normalised mantissa = 0x%016lx\n", mant);
  printf("High bits of rest of mantissa = 0x%016lx\n", highbitsrr);
  printf("Exponent to be inserted on mantissa = rexp = %d\n", rexp);
#endif

  /* Put the result exponent rexp onto the mantissa pattern */
  u = ((__UINT8_T)rexp + EXPBIAS_DP64) << EXPSHIFTBITS_DP64;
  ux = (mant & MANTBITS_DP64) | u;
  if (determ)
    /* If we negated the mantissa we negate x too */
    ux |= SIGNBIT_DP64;
  PUT_BITS_DP64(ux, x);

  /* Create the bit pattern for rr */
  highbitsrr >>= 12; /* Note this is shifted one place too far */
  u = ((__UINT8_T)rexp + EXPBIAS_DP64 - 53) << EXPSHIFTBITS_DP64;
  PUT_BITS_DP64(u, t);
  u |= highbitsrr;
  PUT_BITS_DP64(u, xx);

  /* Subtract the implicit bit we accidentally added */
  xx -= t;
  /* Set the correct sign, and double to account for the
     "one place too far" shift */
  if (determ)
    xx *= -2.0;
  else
    xx *= 2.0;

#ifdef DEBUGGING_PRINT
  printf("(lead part of x*2/pi) = %25.20e = %s\n", x, double2hex(&x));
  printf("(tail part of x*2/pi) = %25.20e = %s\n", xx, double2hex(&xx));
#endif

  /* (x,xx) is an extra-precise version of the fractional part of
     x * 2 / pi. Multiply (x,xx) by pi/2 in extra precision
     to get the reduced argument (r,rr). */
  {
    double hx, tx, c, cc;
    /* Split x into hx (head) and tx (tail) */
    GET_BITS_DP64(x, ux);
    ux &= 0xfffffffff8000000;
    PUT_BITS_DP64(ux, hx);
    tx = x - hx;

    c = piby2_lead * x;
    cc = ((((piby2_part1 * hx - c) + piby2_part1 * tx) + piby2_part2 * hx) +
          piby2_part2 * tx) +
         (piby2_lead * xx + piby2_part3 * x);
    *r = c + cc;
    *rr = (c - *r) + cc;
  }

#ifdef DEBUGGING_PRINT
  printf(" (r,rr) = lead and tail parts of frac(x*2/pi) * pi/2:\n");
  printf(" r = %25.20e = %s\n", *r, double2hex(r));
  printf("rr = %25.20e = %s\n", *rr, double2hex(rr));
  printf("region = (number of pi/2 subtracted from x) mod 4 = %d\n", *region);
#endif
  return;
}
#endif /* USE_REMAINDER_PIBY_INLINE */

#endif /* LIBM_INLINES_AMD_H_INCLUDED */
