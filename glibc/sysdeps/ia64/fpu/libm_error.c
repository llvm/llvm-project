/* file: libm_error.c */


// Copyright (c) 2000 - 2005, Intel Corporation
// All rights reserved.
//
// Contributed 2000 by the Intel Numerics Group, Intel Corporation
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
// * Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//
// * Redistributions in binary form must reproduce the above copyright
// notice, this list of conditions and the following disclaimer in the
// documentation and/or other materials provided with the distribution.
//
// * The name of Intel Corporation may not be used to endorse or promote
// products derived from this software without specific prior written
// permission.

//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
// A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL INTEL OR ITS
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
// OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Intel Corporation is the author of this code, and requests that all
// problem reports or change requests be submitted to it directly at
// http://www.intel.com/software/products/opensource/libraries/num.htm.
//
// History
//==============================================================
//  2/02/00: Initial version
//  3/22/00: Updated to support flexible and dynamic error handling.
//  8/16/00: Changed all matherr function-calls to use the pmatherr
//           function-pointers.
// 10/03/00: Corrected a scalb type.
// 11/28/00: Changed INPUT_XL to INPUT_XD for scalb_underflow case.
// 12/07/00: Added code to make scalbn error support equivalent to ldexp.
//  2/07/01: Added __declspec(align(16)) to long double constants to correct
//           alignment problem.
//  4/23/01: Added code for remquo
//  6/07/01: Added code for fdim, lrint, lround, llrint, llround
//           Deleted code for remquo
//  8/15/01: Added code for scalbln, nexttoward
// 12/10/01: Added code for erfc
// 12/27/01: Added code for degree argument functions
// 01/02/02: Added code for tand, cotd
// 01/15/02: Corrected SVID/XOPEN code for log1p, pow, and acosh
// 01/25/02: Corrected ISOC for lgamma and gamma to return EDOM for neg ints
// 01/28/02: Corrected SVID/XOPEN stderr message for log2
// 05/20/02: Added code for cot
// 07/01/02: Added code for sinhcosh
// 10/04/02: Underflow detection in ISOC path redefined to
//           be zero rather than tiny and inexact
// 12/06/02: Added code for annuity and compound
// 01/30/03: Corrected test for underflow in ISOC path to not set denormal
// 04/10/03: Corrected ISOC branch for gamma/lgamma to return ERANGE for neg ints.
//           Added code for tgamma
// 04/11/03: Corrected POSIX/SVID/XOPEN branches for gamma/lgamma
//           to return EDOM for neg ints.
// 09/08/03: Corrected XOPEN/SVID result for pow overflow with neg x, pos y.
// 10/14/03: Added ILP32 ifdef
// 12/12/03: Corrected XOPEN/SVID results for powf_zero_to_negative,
//           powl_neg_to_non_integer, atan2f_zero, atan2df_zero,
//           acoshf_lt_one, acosh_lt_one.
// 12/07/04: Cast name strings as char *.
// 12/08/04: Corrected POSIX behavior for atan2_zero, acos_gt_one, asin_gt_one,
//           log_negative, log10_negative, log1p_negative, and log2_negative.
//           Added SVID and XOPEN case log2l_zero.
// 12/13/04: Corrected POSIX behavior for exp2_overflow, exp2_underflow,
//           exp10_overflow, exp10_underflow.  Added ISOC to set errno for
//           exp10_underflow.
// 12/14/04: Corrected POSIX behavior for nextafter_overflow,
//           nextafter_underflow, nexttoward_overflow, nexttoward_underflow.
//           Added ISOC to set errno for nextafter and nexttoward underflow.
// 12/15/04: Corrected POSIX behavior for exp, exp2, and exp10 underflow.
// 03/31/05: Added missing ALIGNIT statement to 6 float constants.

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include "libm_support.h"
#include <math-svid-compat.h>

#ifdef _LIBC
# define pmatherr matherr
# define pmatherrf matherrf
# define pmatherrl matherrl
# if LIBM_SVID_COMPAT
compat_symbol_reference (libm, matherrf, matherrf, GLIBC_2_2_3);
compat_symbol_reference (libm, matherrl, matherrl, GLIBC_2_2_3);
# else
#  define matherrf(EXC) ((void) (EXC), 0)
#  define matherrl(EXC) ((void) (EXC), 0)
# endif
#else
_LIB_VERSION_TYPE
#if defined( __POSIX__ )
_LIB_VERSIONIMF = _POSIX_;
#elif defined( __XOPEN__ )
_LIB_VERSIONIMF = _XOPEN_;
#elif defined( __SVID__ )
_LIB_VERSIONIMF = _SVID_;
#elif defined( __IEEE__ )
_LIB_VERSIONIMF = _IEEE_;
#else
_LIB_VERSIONIMF = _ISOC_;
#endif

/************************************************************/
/* matherrX function pointers and setusermatherrX functions */
/************************************************************/
int (*pmatherrf)(struct exceptionf*) = MATHERR_F;
int (*pmatherr)(struct EXC_DECL_D*) = MATHERR_D;
int (*pmatherrl)(struct exceptionl*) = matherrl;

void __libm_setusermatherrf( int(*user_merrf)(struct exceptionf*) )
{   pmatherrf = ( (user_merrf==NULL)? (MATHERR_F) : (user_merrf) ); }

void __libm_setusermatherr( int(*user_merr)(struct EXC_DECL_D*) )
{   pmatherr = ( (user_merr==NULL)? (MATHERR_D) : (user_merr) );    }

void __libm_setusermatherrl( int(*user_merrl)(struct exceptionl*) )
{   pmatherrl = ( (user_merrl==NULL)? (matherrl) : (user_merrl) );  }

#endif /* !_LIBC */

/***********************************************/
/* error-handling function, libm_error_support */
/***********************************************/
void __libm_error_support(void *arg1,void *arg2,void *retval,error_types input_tag)
{

# ifdef __cplusplus
struct __exception exc;
# else
struct exception  exc;
# endif

struct exceptionf excf;
struct exceptionl excl;

# ifdef __GNUC__
#define ALIGNIT __attribute__ ((__aligned__ (16)))
# elif defined opensource
#define ALIGNIT
# else
#define ALIGNIT __declspec(align(16))
# endif

# ifdef SIZE_LONG_INT_64
#define __INT_64__ signed long
# else
# if ILP32
#define __INT_64__ signed long long
# else
#define __INT_64__ __int64
# endif
# endif


#define _DECL_NUM(type, prefix, var, bytes...)	\
  ALIGNIT static const union {			\
    const char _bytes[sizeof (type)];		\
    const type num;				\
  } prefix ## var = {				\
    ._bytes = bytes,				\
  }

#define DECL_FLOAT(var, bytes...) \
  _DECL_NUM (float, float_, var, ##bytes)

DECL_FLOAT(inf,      {0x00,0x00,0x80,0x7F});
DECL_FLOAT(huge,     {0xFF,0xFF,0x7F,0x7F});
DECL_FLOAT(zero,     {0x00,0x00,0x00,0x00});
DECL_FLOAT(neg_inf,  {0x00,0x00,0x80,0xFF});
DECL_FLOAT(neg_huge, {0xFF,0xFF,0x7F,0xFF});
DECL_FLOAT(neg_zero, {0x00,0x00,0x00,0x80});

#define DECL_DOUBLE(var, bytes...) \
  _DECL_NUM (double, double_, var, ##bytes)

DECL_DOUBLE(inf,      {0x00,0x00,0x00,0x00,0x00,0x00,0xF0,0x7F});
#ifndef _LIBC
DECL_DOUBLE(huge,     {0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xEF,0x7F});
#endif
DECL_DOUBLE(zero,     {0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00});
DECL_DOUBLE(neg_inf,  {0x00,0x00,0x00,0x00,0x00,0x00,0xF0,0xFF});
#ifndef _LIBC
DECL_DOUBLE(neg_huge, {0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xEF,0xFF});
#endif
DECL_DOUBLE(neg_zero, {0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x80});

#define DECL_LONG_DOUBLE(var, bytes...) \
  _DECL_NUM (long double, long_double_, var, ##bytes)

DECL_LONG_DOUBLE(inf,      {0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x80,0xFF,0x7F,0x00,0x00,0x00,0x00,0x00,0x00});
#ifndef _LIBC
DECL_LONG_DOUBLE(huge,     {0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFE,0x7F,0x00,0x00,0x00,0x00,0x00,0x00});
#endif
DECL_LONG_DOUBLE(zero,     {0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00});
DECL_LONG_DOUBLE(neg_inf,  {0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x80,0xFF,0xFF,0x00,0x00,0x00,0x00,0x00,0x00});
#ifndef _LIBC
DECL_LONG_DOUBLE(neg_huge, {0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFE,0xFF,0x00,0x00,0x00,0x00,0x00,0x00});
#endif
DECL_LONG_DOUBLE(neg_zero, {0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x80,0x00,0x00,0x00,0x00,0x00,0x00});


#define RETVAL_HUGE_VALL *(long double *)retval =  long_double_inf.num
#define RETVAL_NEG_HUGE_VALL *(long double *)retval = long_double_neg_inf.num
#define RETVAL_HUGEL *(long double *)retval = (long double)float_huge.num
#define RETVAL_NEG_HUGEL *(long double *)retval = (long double)float_neg_huge.num

#define RETVAL_HUGE_VALD *(double *)retval = double_inf.num
#define RETVAL_NEG_HUGE_VALD *(double *)retval = double_neg_inf.num
#define RETVAL_HUGED *(double *)retval = (double)float_huge.num
#define RETVAL_NEG_HUGED *(double *)retval = (double)float_neg_huge.num

#define RETVAL_HUGE_VALF *(float *)retval =  float_inf.num
#define RETVAL_NEG_HUGE_VALF *(float *)retval = float_neg_inf.num
#define RETVAL_HUGEF *(float *)retval = float_huge.num
#define RETVAL_NEG_HUGEF *(float *)retval = float_neg_huge.num

#define ZEROL_VALUE long_double_zero.num
#define ZEROD_VALUE double_zero.num
#define ZEROF_VALUE float_zero.num

#define RETVAL_ZEROL *(long double *)retval = long_double_zero.num
#define RETVAL_ZEROD *(double *)retval = double_zero.num
#define RETVAL_ZEROF *(float *)retval = float_zero.num

#define RETVAL_NEG_ZEROL *(long double *)retval = long_double_neg_zero.num
#define RETVAL_NEG_ZEROD *(double *)retval = double_neg_zero.num
#define RETVAL_NEG_ZEROF *(float *)retval = float_neg_zero.num

#define RETVAL_ONEL *(long double *)retval = (long double) 1.0
#define RETVAL_ONED *(double *)retval = 1.0
#define RETVAL_ONEF *(float *)retval = 1.0f

#define NOT_MATHERRL excl.arg1=*(long double *)arg1;excl.arg2=*(long double *)arg2;excl.retval=*(long double *)retval;if(!pmatherrl(&excl))
#define NOT_MATHERRD exc.arg1=*(double *)arg1;exc.arg2=*(double *)arg2;exc.retval=*(double *)retval;if(!pmatherr(&exc))
#define NOT_MATHERRF excf.arg1=*(float *)arg1;excf.arg2=*(float *)arg2;excf.retval=*(float *)retval;if(!pmatherrf(&excf))

#define ifSVID if(_LIB_VERSIONIMF==_SVID_)

#define NAMEL excl.name
#define NAMED exc.name
#define NAMEF excf.name

//
// These should work OK for MS because they are ints -
// leading underbars are not necessary.
//

#define DOMAIN          1
#define SING            2
#define OVERFLOW        3
#define UNDERFLOW       4
#define TLOSS           5
#define PLOSS           6

#define SINGL excl.type = SING
#define DOMAINL excl.type = DOMAIN
#define OVERFLOWL excl.type = OVERFLOW
#define UNDERFLOWL excl.type = UNDERFLOW
#define TLOSSL excl.type = TLOSS
#define SINGD exc.type = SING
#define DOMAIND exc.type = DOMAIN
#define OVERFLOWD exc.type = OVERFLOW
#define UNDERFLOWD exc.type = UNDERFLOW
#define TLOSSD exc.type = TLOSS
#define SINGF excf.type = SING
#define DOMAINF excf.type = DOMAIN
#define OVERFLOWF excf.type = OVERFLOW
#define UNDERFLOWF excf.type = UNDERFLOW
#define TLOSSF excf.type = TLOSS

#define INPUT_XL (excl.arg1=*(long double*)arg1)
#define INPUT_XD (exc.arg1=*(double*)arg1)
#define INPUT_XF (excf.arg1=*(float*)arg1)
#define INPUT_YL (excl.arg2=*(long double*)arg2)
#define INPUT_YD (exc.arg2=*(double*)arg2)
#define INPUT_YF (excf.arg2=*(float*)arg2)
#define INPUT_RESL (*(long double *)retval)
#define INPUT_RESD (*(double *)retval)
#define INPUT_RESF (*(float *)retval)
#define INPUT_RESI64 (*(__INT_64__ *)retval)

#define WRITEL_LOG_ZERO fputs("logl: SING error\n",stderr)
#define WRITED_LOG_ZERO fputs("log: SING error\n",stderr)
#define WRITEF_LOG_ZERO fputs("logf: SING error\n",stderr)
#define WRITEL_LOG_NEGATIVE fputs("logl: DOMAIN error\n",stderr)
#define WRITED_LOG_NEGATIVE fputs("log: DOMAIN error\n",stderr)
#define WRITEF_LOG_NEGATIVE fputs("logf: DOMAIN error\n",stderr)
#define WRITEL_Y0_ZERO fputs("y0l: DOMAIN error\n",stderr)
#define WRITED_Y0_ZERO fputs("y0: DOMAIN error\n",stderr)
#define WRITEF_Y0_ZERO fputs("y0f: DOMAIN error\n",stderr)
#define WRITEL_Y0_NEGATIVE fputs("y0l: DOMAIN error\n",stderr)
#define WRITED_Y0_NEGATIVE fputs("y0: DOMAIN error\n",stderr)
#define WRITEF_Y0_NEGATIVE fputs("y0f: DOMAIN error\n",stderr)
#define WRITEL_Y1_ZERO fputs("y1l: DOMAIN error\n",stderr)
#define WRITED_Y1_ZERO fputs("y1: DOMAIN error\n",stderr)
#define WRITEF_Y1_ZERO fputs("y1f: DOMAIN error\n",stderr)
#define WRITEL_Y1_NEGATIVE fputs("y1l: DOMAIN error\n",stderr)
#define WRITED_Y1_NEGATIVE fputs("y1: DOMAIN error\n",stderr)
#define WRITEF_Y1_NEGATIVE fputs("y1f: DOMAIN error\n",stderr)
#define WRITEL_YN_ZERO fputs("ynl: DOMAIN error\n",stderr)
#define WRITED_YN_ZERO fputs("yn: DOMAIN error\n",stderr)
#define WRITEF_YN_ZERO fputs("ynf: DOMAIN error\n",stderr)
#define WRITEL_YN_NEGATIVE fputs("ynl: DOMAIN error\n",stderr)
#define WRITED_YN_NEGATIVE fputs("yn: DOMAIN error\n",stderr)
#define WRITEF_YN_NEGATIVE fputs("ynf: DOMAIN error\n",stderr)
#define WRITEL_LOG1P_ZERO fputs("log1pl: SING error\n",stderr)
#define WRITED_LOG1P_ZERO fputs("log1p: SING error\n",stderr)
#define WRITEF_LOG1P_ZERO fputs("log1pf: SING error\n",stderr)
#define WRITEL_LOG1P_NEGATIVE fputs("log1pl: DOMAIN error\n",stderr)
#define WRITED_LOG1P_NEGATIVE fputs("log1p: DOMAIN error\n",stderr)
#define WRITEF_LOG1P_NEGATIVE fputs("log1pf: DOMAIN error\n",stderr)
#define WRITEL_LOG10_ZERO fputs("log10l: SING error\n",stderr)
#define WRITED_LOG10_ZERO fputs("log10: SING error\n",stderr)
#define WRITEF_LOG10_ZERO fputs("log10f: SING error\n",stderr)
#define WRITEL_LOG10_NEGATIVE fputs("log10l: DOMAIN error\n",stderr)
#define WRITED_LOG10_NEGATIVE fputs("log10: DOMAIN error\n",stderr)
#define WRITEF_LOG10_NEGATIVE fputs("log10f: DOMAIN error\n",stderr)
#define WRITEL_LOG2_ZERO fputs("log2l: SING error\n",stderr)
#define WRITED_LOG2_ZERO fputs("log2: SING error\n",stderr)
#define WRITEF_LOG2_ZERO fputs("log2f: SING error\n",stderr)
#define WRITEL_LOG2_NEGATIVE fputs("log2l: DOMAIN error\n",stderr)
#define WRITED_LOG2_NEGATIVE fputs("log2: DOMAIN error\n",stderr)
#define WRITEF_LOG2_NEGATIVE fputs("log2f: DOMAIN error\n",stderr)
#define WRITEL_POW_ZERO_TO_ZERO fputs("powl(0,0): DOMAIN error\n",stderr)
#define WRITED_POW_ZERO_TO_ZERO fputs("pow(0,0): DOMAIN error\n",stderr)
#define WRITEF_POW_ZERO_TO_ZERO fputs("powf(0,0): DOMAIN error\n",stderr)
#define WRITEL_POW_ZERO_TO_NEGATIVE fputs("powl(0,negative): DOMAIN error\n",stderr)
#define WRITED_POW_ZERO_TO_NEGATIVE fputs("pow(0,negative): DOMAIN error\n",stderr)
#define WRITEF_POW_ZERO_TO_NEGATIVE fputs("powf(0,negative): DOMAIN error\n",stderr)
#define WRITEL_POW_NEG_TO_NON_INTEGER fputs("powl(negative,non-integer): DOMAIN error\n",stderr)
#define WRITED_POW_NEG_TO_NON_INTEGER fputs("pow(negative,non-integer): DOMAIN error\n",stderr)
#define WRITEF_POW_NEG_TO_NON_INTEGER fputs("powf(negative,non-integer): DOMAIN error\n",stderr)
#define WRITEL_ATAN2_ZERO_BY_ZERO fputs("atan2l: DOMAIN error\n",stderr)
#define WRITED_ATAN2_ZERO_BY_ZERO fputs("atan2: DOMAIN error\n",stderr)
#define WRITEF_ATAN2_ZERO_BY_ZERO fputs("atan2f: DOMAIN error\n",stderr)
#define WRITEL_SQRT fputs("sqrtl: DOMAIN error\n",stderr)
#define WRITED_SQRT fputs("sqrt: DOMAIN error\n",stderr)
#define WRITEF_SQRT fputs("sqrtf: DOMAIN error\n",stderr)
#define WRITEL_FMOD fputs("fmodl: DOMAIN error\n",stderr)
#define WRITED_FMOD fputs("fmod: DOMAIN error\n",stderr)
#define WRITEF_FMOD fputs("fmodf: DOMAIN error\n",stderr)
#define WRITEL_REM fputs("remainderl: DOMAIN error\n",stderr)
#define WRITED_REM fputs("remainder: DOMAIN error\n",stderr)
#define WRITEF_REM fputs("remainderf: DOMAIN error\n",stderr)
#define WRITEL_ACOS fputs("acosl: DOMAIN error\n",stderr)
#define WRITED_ACOS fputs("acos: DOMAIN error\n",stderr)
#define WRITEF_ACOS fputs("acosf: DOMAIN error\n",stderr)
#define WRITEL_ASIN fputs("asinl: DOMAIN error\n",stderr)
#define WRITED_ASIN fputs("asin: DOMAIN error\n",stderr)
#define WRITEF_ASIN fputs("asinf: DOMAIN error\n",stderr)
#define WRITEL_ACOSH fputs("acoshl: DOMAIN error\n",stderr)
#define WRITED_ACOSH fputs("acosh: DOMAIN error\n",stderr)
#define WRITEF_ACOSH fputs("acoshf: DOMAIN error\n",stderr)
#define WRITEL_ATANH_GT_ONE fputs("atanhl: DOMAIN error\n",stderr)
#define WRITED_ATANH_GT_ONE fputs("atanh: DOMAIN error\n",stderr)
#define WRITEF_ATANH_GT_ONE fputs("atanhf: DOMAIN error\n",stderr)
#define WRITEL_ATANH_EQ_ONE fputs("atanhl: SING error\n",stderr)
#define WRITED_ATANH_EQ_ONE fputs("atanh: SING error\n",stderr)
#define WRITEF_ATANH_EQ_ONE fputs("atanhf: SING error\n",stderr)
#define WRITEL_LGAMMA_NEGATIVE fputs("lgammal: SING error\n",stderr)
#define WRITED_LGAMMA_NEGATIVE fputs("lgamma: SING error\n",stderr)
#define WRITEF_LGAMMA_NEGATIVE fputs("lgammaf: SING error\n",stderr)
#define WRITEL_GAMMA_NEGATIVE fputs("gammal: SING error\n",stderr)
#define WRITED_GAMMA_NEGATIVE fputs("gamma: SING error\n",stderr)
#define WRITEF_GAMMA_NEGATIVE fputs("gammaf: SING error\n",stderr)
#define WRITEL_TGAMMA_NEGATIVE fputs("tgammal: SING error\n",stderr)
#define WRITED_TGAMMA_NEGATIVE fputs("tgamma: SING error\n",stderr)
#define WRITEF_TGAMMA_NEGATIVE fputs("tgammaf: SING error\n",stderr)
#define WRITEL_J0_TLOSS  fputs("j0l: TLOSS error\n",stderr)
#define WRITEL_Y0_TLOSS  fputs("y0l: TLOSS error\n",stderr)
#define WRITEL_J1_TLOSS  fputs("j1l: TLOSS error\n",stderr)
#define WRITEL_Y1_TLOSS  fputs("y1l: TLOSS error\n",stderr)
#define WRITEL_JN_TLOSS  fputs("jnl: TLOSS error\n",stderr)
#define WRITEL_YN_TLOSS  fputs("ynl: TLOSS error\n",stderr)
#define WRITED_J0_TLOSS  fputs("j0: TLOSS error\n",stderr)
#define WRITED_Y0_TLOSS  fputs("y0: TLOSS error\n",stderr)
#define WRITED_J1_TLOSS  fputs("j1: TLOSS error\n",stderr)
#define WRITED_Y1_TLOSS  fputs("y1: TLOSS error\n",stderr)
#define WRITED_JN_TLOSS  fputs("jn: TLOSS error\n",stderr)
#define WRITED_YN_TLOSS  fputs("yn: TLOSS error\n",stderr)
#define WRITEF_J0_TLOSS  fputs("j0f: TLOSS error\n",stderr)
#define WRITEF_Y0_TLOSS  fputs("y0f: TLOSS error\n",stderr)
#define WRITEF_J1_TLOSS  fputs("j1f: TLOSS error\n",stderr)
#define WRITEF_Y1_TLOSS  fputs("y1f: TLOSS error\n",stderr)
#define WRITEF_JN_TLOSS  fputs("jnf: TLOSS error\n",stderr)
#define WRITEF_YN_TLOSS  fputs("ynf: TLOSS error\n",stderr)
#define WRITEL_ACOSD fputs("acosdl: DOMAIN error\n",stderr)
#define WRITED_ACOSD fputs("acosd: DOMAIN error\n",stderr)
#define WRITEF_ACOSD fputs("acosdf: DOMAIN error\n",stderr)
#define WRITEL_ASIND fputs("asindl: DOMAIN error\n",stderr)
#define WRITED_ASIND fputs("asind: DOMAIN error\n",stderr)
#define WRITEF_ASIND fputs("asindf: DOMAIN error\n",stderr)
#define WRITEL_ATAN2D_ZERO_BY_ZERO fputs("atan2dl: DOMAIN error\n",stderr)
#define WRITED_ATAN2D_ZERO_BY_ZERO fputs("atan2d: DOMAIN error\n",stderr)
#define WRITEF_ATAN2D_ZERO_BY_ZERO fputs("atan2df: DOMAIN error\n",stderr)


/***********************/
/* IEEE Path           */
/***********************/
if(_LIB_VERSIONIMF==_IEEE_) return;

/***********************/
/* C9X Path           */
/***********************/
else if(_LIB_VERSIONIMF==_ISOC_)
{
  switch(input_tag)
  {
    case logl_zero:
    case log_zero:
    case logf_zero:
    case log10l_zero:
    case log10_zero:
    case log10f_zero:
    case log2l_zero:
    case log2_zero:
    case log2f_zero:
    case log1pl_zero:
    case log1p_zero:
    case log1pf_zero:
    case powl_overflow:
    case pow_overflow:
    case powf_overflow:
    case expl_overflow:
    case exp_overflow:
    case expf_overflow:
    case exp2l_overflow:
    case exp2_overflow:
    case exp2f_overflow:
    case exp10l_overflow:
    case exp10_overflow:
    case exp10f_overflow:
    case expm1l_overflow:
    case expm1_overflow:
    case expm1f_overflow:
    case hypotl_overflow:
    case hypot_overflow:
    case hypotf_overflow:
    case sinhl_overflow:
    case sinh_overflow:
    case sinhf_overflow:
    case atanhl_eq_one:
    case atanh_eq_one:
    case atanhf_eq_one:
    case scalbl_overflow:
    case scalb_overflow:
    case scalbf_overflow:
    case coshl_overflow:
    case cosh_overflow:
    case coshf_overflow:
    case nextafterl_overflow:
    case nextafter_overflow:
    case nextafterf_overflow:
    case nextafterl_underflow:
    case nextafter_underflow:
    case nextafterf_underflow:
    case nexttowardl_overflow:
    case nexttoward_overflow:
    case nexttowardf_overflow:
    case nexttowardl_underflow:
    case nexttoward_underflow:
    case nexttowardf_underflow:
    case scalbnl_overflow:
    case scalbn_overflow:
    case scalbnf_overflow:
    case scalblnl_overflow:
    case scalbln_overflow:
    case scalblnf_overflow:
    case ldexpl_overflow:
    case ldexp_overflow:
    case ldexpf_overflow:
    case lgammal_overflow:
    case lgamma_overflow:
    case lgammaf_overflow:
    case gammal_overflow:
    case gamma_overflow:
    case gammaf_overflow:
    case lgammal_negative:
    case lgamma_negative:
    case lgammaf_negative:
    case gammal_negative:
    case gamma_negative:
    case gammaf_negative:
    case ilogbl_zero:
    case ilogb_zero:
    case ilogbf_zero:
    case fdiml_overflow:
    case fdim_overflow:
    case fdimf_overflow:
    case llrintl_large:
    case llrint_large:
    case llrintf_large:
    case llroundl_large:
    case llround_large:
    case llroundf_large:
    case lrintl_large:
    case lrint_large:
    case lrintf_large:
    case lroundl_large:
    case lround_large:
    case lroundf_large:
    case tandl_overflow:
    case tand_overflow:
    case tandf_overflow:
    case cotdl_overflow:
    case cotd_overflow:
    case cotdf_overflow:
    case cotl_overflow:
    case cot_overflow:
    case cotf_overflow:
    case sinhcoshl_overflow:
    case sinhcosh_overflow:
    case sinhcoshf_overflow:
    case annuityl_overflow:
    case annuity_overflow:
    case annuityf_overflow:
    case compoundl_overflow:
    case compound_overflow:
    case compoundf_overflow:
    case tgammal_overflow:
    case tgamma_overflow:
    case tgammaf_overflow:
    {
         ERRNO_RANGE; break;
    }
    case powl_underflow:
    case expl_underflow:
    case exp10l_underflow:
    case exp2l_underflow:
    case scalbl_underflow:
    case scalbnl_underflow:
    case scalblnl_underflow:
    case ldexpl_underflow:
    case erfcl_underflow:
    case annuityl_underflow:
    case compoundl_underflow:
    {
       /* Test for zero by testing 64 significand bits for zero. An integer
          test is needed so denormal flag is not set by a floating-point test */
       if ( INPUT_RESI64 == 0 ) ERRNO_RANGE;
       break;
    }
    case pow_underflow:
    case exp_underflow:
    case exp10_underflow:
    case exp2_underflow:
    case scalb_underflow:
    case scalbn_underflow:
    case scalbln_underflow:
    case ldexp_underflow:
    case erfc_underflow:
    case annuity_underflow:
    case compound_underflow:
    {
       /* Test for zero by testing exp and significand bits for zero. An integer
          test is needed so denormal flag is not set by a floating-point test */
       if ( (INPUT_RESI64 << 1) == 0 ) ERRNO_RANGE;
       break;
    }
    case powf_underflow:
    case expf_underflow:
    case exp10f_underflow:
    case exp2f_underflow:
    case scalbf_underflow:
    case scalbnf_underflow:
    case scalblnf_underflow:
    case ldexpf_underflow:
    case erfcf_underflow:
    case annuityf_underflow:
    case compoundf_underflow:
    {
       /* Test for zero by testing exp and significand bits for zero. An integer
          test is needed so denormal flag is not set by a floating-point test */
       if ( (INPUT_RESI64 << 33) == 0 ) ERRNO_RANGE;
       break;
    }
    case logl_negative:
    case log_negative:
    case logf_negative:
    case log10l_negative:
    case log10_negative:
    case log10f_negative:
    case log2l_negative:
    case log2_negative:
    case log2f_negative:
    case log1pl_negative:
    case log1p_negative:
    case log1pf_negative:
    case sqrtl_negative:
    case sqrt_negative:
    case sqrtf_negative:
    case atan2l_zero:
    case atan2_zero:
    case atan2f_zero:
    case powl_zero_to_negative:
    case powl_neg_to_non_integer:
    case pow_zero_to_negative:
    case pow_neg_to_non_integer:
    case powf_zero_to_negative:
    case powf_neg_to_non_integer:
    case fmodl_by_zero:
    case fmod_by_zero:
    case fmodf_by_zero:
    case atanhl_gt_one:
    case atanh_gt_one:
    case atanhf_gt_one:
    case acosl_gt_one:
    case acos_gt_one:
    case acosf_gt_one:
    case asinl_gt_one:
    case asin_gt_one:
    case asinf_gt_one:
    case logbl_zero:
    case logb_zero:
    case logbf_zero:
    case acoshl_lt_one:
    case acosh_lt_one:
    case acoshf_lt_one:
    case y0l_zero:
    case y0_zero:
    case y0f_zero:
    case y1l_zero:
    case y1_zero:
    case y1f_zero:
    case ynl_zero:
    case yn_zero:
    case ynf_zero:
    case y0l_negative:
    case y0_negative:
    case y0f_negative:
    case y1l_negative:
    case y1_negative:
    case y1f_negative:
    case ynl_negative:
    case yn_negative:
    case ynf_negative:
    case acosdl_gt_one:
    case acosd_gt_one:
    case acosdf_gt_one:
    case asindl_gt_one:
    case asind_gt_one:
    case asindf_gt_one:
    case atan2dl_zero:
    case atan2d_zero:
    case atan2df_zero:
    case annuityl_by_zero:
    case annuity_by_zero:
    case annuityf_by_zero:
    case annuityl_less_m1:
    case annuity_less_m1:
    case annuityf_less_m1:
    case compoundl_by_zero:
    case compound_by_zero:
    case compoundf_by_zero:
    case compoundl_less_m1:
    case compound_less_m1:
    case compoundf_less_m1:
    case tgammal_negative:
    case tgamma_negative:
    case tgammaf_negative:
    {
         ERRNO_DOMAIN; break;
    }
    default:
      break;
   }
   return;
}

/***********************/
/* _POSIX_ Path        */
/***********************/

else if(_LIB_VERSIONIMF==_POSIX_)
{
switch(input_tag)
  {
  case gammal_overflow:
  case lgammal_overflow:
  case tgammal_overflow:
  {
       RETVAL_HUGE_VALL; ERRNO_RANGE; break;
  }
  case gamma_overflow:
  case lgamma_overflow:
  case tgamma_overflow:
  {
       RETVAL_HUGE_VALD; ERRNO_RANGE; break;
  }
  case gammaf_overflow:
  case lgammaf_overflow:
  case tgammaf_overflow:
  {
       RETVAL_HUGE_VALF; ERRNO_RANGE; break;
  }
  case gammal_negative:
  case gamma_negative:
  case gammaf_negative:
  case lgammal_negative:
  case lgamma_negative:
  case lgammaf_negative:
  case tgammal_negative:
  case tgamma_negative:
  case tgammaf_negative:
  {
       ERRNO_DOMAIN; break;
  }
  case ldexpl_overflow:
  case ldexpl_underflow:
  case ldexp_overflow:
  case ldexp_underflow:
  case ldexpf_overflow:
  case ldexpf_underflow:
  case scalbnl_overflow:
  case scalbnl_underflow:
  case scalbn_overflow:
  case scalbn_underflow:
  case scalbnf_overflow:
  case scalbnf_underflow:
  case scalblnl_overflow:
  case scalblnl_underflow:
  case scalbln_overflow:
  case scalbln_underflow:
  case scalblnf_overflow:
  case scalblnf_underflow:
  case tandl_overflow:
  case tand_overflow:
  case tandf_overflow:
  case cotdl_overflow:
  case cotd_overflow:
  case cotdf_overflow:
  case cotl_overflow:
  case cot_overflow:
  case cotf_overflow:
  case sinhcoshl_overflow:
  case sinhcosh_overflow:
  case sinhcoshf_overflow:
  case nextafterl_overflow:
  case nextafter_overflow:
  case nextafterf_overflow:
  case nextafterl_underflow:
  case nextafter_underflow:
  case nextafterf_underflow:
  case nexttowardl_overflow:
  case nexttoward_overflow:
  case nexttowardf_overflow:
  case nexttowardl_underflow:
  case nexttoward_underflow:
  case nexttowardf_underflow:
  {
       ERRNO_RANGE; break;
  }
  case atanhl_gt_one:
  case atanhl_eq_one:
    /* atanhl(|x| >= 1) */
    {
       ERRNO_DOMAIN; break;
    }
  case atanh_gt_one:
  case atanh_eq_one:
    /* atanh(|x| >= 1) */
    {
       ERRNO_DOMAIN; break;
    }
  case atanhf_gt_one:
  case atanhf_eq_one:
    /* atanhf(|x| >= 1) */
    {
       ERRNO_DOMAIN; break;
    }
  case sqrtl_negative:
    /* sqrtl(x < 0) */
    {
       ERRNO_DOMAIN; break;
    }
  case sqrt_negative:
    /* sqrt(x < 0) */
    {
       ERRNO_DOMAIN; break;
    }
  case sqrtf_negative:
    /* sqrtf(x < 0) */
    {
       ERRNO_DOMAIN; break;
    }
  case y0l_zero:
  case y1l_zero:
  case ynl_zero:
    /* y0l(0) */
    /* y1l(0) */
    /* ynl(0) */
    {
       RETVAL_NEG_HUGE_VALL; ERRNO_DOMAIN; break;
    }
  case y0_zero:
  case y1_zero:
  case yn_zero:
    /* y0(0) */
    /* y1(0) */
    /* yn(0) */
    {
       RETVAL_NEG_HUGE_VALD; ERRNO_DOMAIN; break;
    }
  case y0f_zero:
  case y1f_zero:
  case ynf_zero:
    /* y0f(0) */
    /* y1f(0) */
    /* ynf(0) */
    {
       RETVAL_NEG_HUGE_VALF; ERRNO_DOMAIN; break;
    }
  case y0l_negative:
  case y1l_negative:
  case ynl_negative:
    /* y0l(x < 0) */
    /* y1l(x < 0) */
    /* ynl(x < 0) */
    {
#ifndef _LIBC
       RETVAL_NEG_HUGE_VALL;
#endif
       ERRNO_DOMAIN; break;
    }
  case y0_negative:
  case y1_negative:
  case yn_negative:
    /* y0(x < 0) */
    /* y1(x < 0) */
    /* yn(x < 0) */
    {
       RETVAL_NEG_HUGE_VALD; ERRNO_DOMAIN; break;
    }
  case y0f_negative:
  case y1f_negative:
  case ynf_negative:
    /* y0f(x < 0) */
    /* y1f(x < 0) */
    /* ynf(x < 0) */
    {
       RETVAL_NEG_HUGE_VALF; ERRNO_DOMAIN; break;
    }
  case logl_zero:
  case log1pl_zero:
  case log10l_zero:
  case log2l_zero:
    /* logl(0) */
    /* log1pl(-1) */
    /* log10l(0) */
    /* log2l(0) */
    {
       RETVAL_NEG_HUGE_VALL; ERRNO_RANGE; break;
    }
  case log_zero:
  case log1p_zero:
  case log10_zero:
  case log2_zero:
   /* log(0) */
   /* log1p(-1) */
   /* log10(0) */
   /* log2(0) */
    {
       RETVAL_NEG_HUGE_VALD; ERRNO_RANGE; break;
    }
  case logf_zero:
  case log1pf_zero:
  case log10f_zero:
  case log2f_zero:
    /* logf(0) */
    /* log1pf(-1) */
    /* log10f(0) */
    /* log2f(0) */
    {
       RETVAL_NEG_HUGE_VALF; ERRNO_RANGE; break;
    }
  case logl_negative:
  case log1pl_negative:
  case log10l_negative:
  case log2l_negative:
    /* logl(x < 0) */
    /* log1pl(x < -1) */
    /* log10l(x < 0) */
    /* log2l(x < 0) */
    {
       ERRNO_DOMAIN; break;
    }
  case log_negative:
  case log1p_negative:
  case log10_negative:
  case log2_negative:
    /* log(x < 0) */
    /* log1p(x < -1) */
    /* log10(x < 0) */
    /* log2(x < 0) */
    {
       ERRNO_DOMAIN; break;
    }
  case logf_negative:
  case log1pf_negative:
  case log10f_negative:
  case log2f_negative:
    /* logf(x < 0) */
    /* log1pf(x < -1) */
    /* log10f(x < 0) */
    /* log2f(x < 0) */
    {
       ERRNO_DOMAIN; break;
    }
  case expl_overflow:
  case exp10l_overflow:
  case exp2l_overflow:
    /* expl overflow */
    /* exp10l overflow */
    /* exp2l overflow */
    {
       RETVAL_HUGE_VALL; ERRNO_RANGE; break;
    }
  case exp_overflow:
  case exp10_overflow:
  case exp2_overflow:
    /* exp overflow */
    /* exp10 overflow */
    /* exp2 overflow */
    {
       RETVAL_HUGE_VALD; ERRNO_RANGE; break;
    }
  case expf_overflow:
  case exp10f_overflow:
  case exp2f_overflow:
    /* expf overflow */
    {
       RETVAL_HUGE_VALF; ERRNO_RANGE; break;
    }
  case expl_underflow:
  case exp10l_underflow:
  case exp2l_underflow:
    /* expl underflow */
    /* exp10l underflow */
    /* exp2l underflow */
    {
       ERRNO_RANGE; break;
    }
  case exp_underflow:
  case exp10_underflow:
  case exp2_underflow:
    /* exp underflow */
    /* exp10 underflow */
    /* exp2 underflow */
    {
       ERRNO_RANGE; break;
    }
  case expf_underflow:
  case exp10f_underflow:
  case exp2f_underflow:
    /* expf underflow */
    /* exp10f underflow */
    /* exp2f underflow */
    {
       ERRNO_RANGE; break;
    }
  case j0l_gt_loss:
  case y0l_gt_loss:
  case j1l_gt_loss:
  case y1l_gt_loss:
  case jnl_gt_loss:
  case ynl_gt_loss:
    /* jn and yn  doubl-extended> XLOSS */
    {
       RETVAL_ZEROL; ERRNO_RANGE; break;
    }
  case j0_gt_loss:
  case y0_gt_loss:
  case j1_gt_loss:
  case y1_gt_loss:
  case jn_gt_loss:
  case yn_gt_loss:
    /* jn and yn double > XLOSS */
    {
       RETVAL_ZEROD; ERRNO_RANGE; break;
    }
  case j0f_gt_loss:
  case y0f_gt_loss:
  case j1f_gt_loss:
  case y1f_gt_loss:
  case jnf_gt_loss:
  case ynf_gt_loss:
    /* j0n and y0n > XLOSS */
    {
       RETVAL_ZEROF; ERRNO_RANGE; break;
    }
  case powl_zero_to_zero:
    /* powl 0**0 */
    {
       break;
    }
  case pow_zero_to_zero:
    /* pow 0**0 */
    {
       break;
    }
  case powf_zero_to_zero:
    /* powf 0**0 */
    {
       break;
    }
  case powl_overflow:
  case annuityl_overflow:
  case compoundl_overflow:
    /* powl(x,y) overflow */
    {
       if (INPUT_RESL < ZEROL_VALUE /*0*/) RETVAL_NEG_HUGE_VALL;
       else RETVAL_HUGE_VALL;
       ERRNO_RANGE; break;
    }
  case pow_overflow:
  case annuity_overflow:
  case compound_overflow:
    /* pow(x,y) overflow */
    {
       if (INPUT_RESD < ZEROD_VALUE /*0*/) RETVAL_NEG_HUGE_VALD;
       else RETVAL_HUGE_VALD;
       ERRNO_RANGE; break;
    }
  case powf_overflow:
  case annuityf_overflow:
  case compoundf_overflow:
    /* powf(x,y) overflow */
    {
       if (INPUT_RESF < ZEROF_VALUE /*0*/) RETVAL_NEG_HUGE_VALF;
       else RETVAL_HUGE_VALF;
       ERRNO_RANGE; break;
    }
  case powl_underflow:
  case annuityl_underflow:
  case compoundl_underflow:
    /* powl(x,y) underflow */
    {
       RETVAL_ZEROL; ERRNO_RANGE; break;
    }
  case pow_underflow:
  case annuity_underflow:
  case compound_underflow:
    /* pow(x,y) underflow */
    {
       RETVAL_ZEROD; ERRNO_RANGE; break;
    }
  case powf_underflow:
  case annuityf_underflow:
  case compoundf_underflow:
    /* powf(x,y) underflow */
    {
       RETVAL_ZEROF; ERRNO_RANGE; break;
    }
  case annuityl_by_zero:
  case annuityl_less_m1:
  case compoundl_by_zero:
  case compoundl_less_m1:
  case annuity_by_zero:
  case annuity_less_m1:
  case compound_by_zero:
  case compound_less_m1:
  case annuityf_by_zero:
  case annuityf_less_m1:
  case compoundf_by_zero:
  case compoundf_less_m1:
    {
       ERRNO_DOMAIN; break;
    }
  case powl_zero_to_negative:
    /* 0**neg */
    {
       ERRNO_DOMAIN; break;
    }
  case pow_zero_to_negative:
    /* 0**neg */
    {
       ERRNO_DOMAIN; break;
    }
  case  powf_zero_to_negative:
    /* 0**neg */
    {
       ERRNO_DOMAIN; break;
    }
  case powl_neg_to_non_integer:
    /* neg**non_integral */
    {
       ERRNO_DOMAIN; break;
    }
  case pow_neg_to_non_integer:
    /* neg**non_integral */
    {
       ERRNO_DOMAIN; break;
    }
  case  powf_neg_to_non_integer:
    /* neg**non-integral */
    {
       ERRNO_DOMAIN; break;
    }
  case  powl_nan_to_zero:
    /* powl(NaN,0.0) */
    /* Special Error */
    {
       break;
    }
  case  pow_nan_to_zero:
    /* pow(NaN,0.0) */
    {
       break;
    }
  case  powf_nan_to_zero:
    /* powf(NaN,0.0) */
    {
       break;
    }
  case atan2l_zero:
  case atan2dl_zero:
    /* atan2l(0,0) */
    /* atan2dl(0,0) */
    {
       break;
    }
  case atan2_zero:
  case atan2d_zero:
    /* atan2(0,0) */
    /* atan2d(0,0) */
    {
       break;
    }
  case atan2f_zero:
  case atan2df_zero:
    /* atan2f(0,0) */
    /* atan2df(0,0) */
    {
       break;
    }
  case expm1l_overflow:
    /* expm1 overflow */
    {
       ERRNO_RANGE; break;
    }
  case expm1_overflow:
    /* expm1 overflow */
    {
       ERRNO_RANGE; break;
    }
  case expm1f_overflow:
    /* expm1f overflow */
    {
       ERRNO_RANGE; break;
    }
  case expm1l_underflow:
    /* expm1 underflow */
    {
       ERRNO_RANGE; break;
    }
  case expm1_underflow:
    /* expm1 underflow */
    {
       ERRNO_RANGE; break;
    }
  case expm1f_underflow:
    /* expm1f underflow */
    {
       ERRNO_RANGE; break;
    }
  case hypotl_overflow:
    /* hypotl overflow */
    {
       RETVAL_HUGE_VALL; ERRNO_RANGE; break;
    }
  case hypot_overflow:
    /* hypot overflow */
    {
       RETVAL_HUGE_VALD; ERRNO_RANGE; break;
    }
  case hypotf_overflow:
    /* hypotf overflow */
    {
       RETVAL_HUGE_VALF; ERRNO_RANGE; break;
    }
  case scalbl_underflow:
    /* scalbl underflow */
    {
       if (INPUT_XL < ZEROL_VALUE /*0*/) RETVAL_NEG_ZEROL;
       else RETVAL_ZEROL;
       ERRNO_RANGE; break;
    }
  case scalb_underflow:
    /* scalb underflow */
    {
       if (INPUT_XD < ZEROD_VALUE /*0*/) RETVAL_NEG_ZEROD;
       else RETVAL_ZEROD;
       ERRNO_RANGE; break;
    }
  case scalbf_underflow:
    /* scalbf underflow */
    {
       if (INPUT_XF < ZEROF_VALUE /*0*/) RETVAL_NEG_ZEROF;
       else RETVAL_ZEROF;
       ERRNO_RANGE; break;
    }
  case scalbl_overflow:
    /* scalbl overflow */
    {
       if (INPUT_XL < ZEROL_VALUE /*0*/) RETVAL_NEG_HUGE_VALL;
       else RETVAL_HUGE_VALL;
       ERRNO_RANGE; break;
    }
  case scalb_overflow:
    /* scalb overflow */
    {
       if (INPUT_XD < ZEROD_VALUE /*0*/) RETVAL_NEG_HUGE_VALD;
       else RETVAL_HUGE_VALD;
       ERRNO_RANGE; break;
    }
  case scalbf_overflow:
    /* scalbf overflow */
    {
       if (INPUT_XF < ZEROF_VALUE /*0*/) RETVAL_NEG_HUGE_VALF;
       else RETVAL_HUGE_VALF;
       ERRNO_RANGE; break;
    }
  case acoshl_lt_one:
    /* acoshl(x < 1) */
    {
       ERRNO_DOMAIN; break;
    }
  case acosh_lt_one:
    /* acosh(x < 1) */
    {
       ERRNO_DOMAIN; break;
    }
  case acoshf_lt_one:
    /* acoshf(x < 1) */
    {
        ERRNO_DOMAIN; break;
    }
  case acosl_gt_one:
  case acosdl_gt_one:
    /* acosl(x > 1) */
    /* acosdl(x > 1) */
    {
       ERRNO_DOMAIN; break;
    }
  case acos_gt_one:
  case acosd_gt_one:
    /* acos(x > 1) */
    /* acosd(x > 1) */
    {
       ERRNO_DOMAIN; break;
    }
  case acosf_gt_one:
  case acosdf_gt_one:
    /* acosf(x > 1) */
    /* acosdf(x > 1) */
    {
       ERRNO_DOMAIN; break;
    }
  case asinl_gt_one:
  case asindl_gt_one:
    /* asinl(x > 1) */
    /* asindl(x > 1) */
    {
       ERRNO_DOMAIN; break;
    }
  case asin_gt_one:
  case asind_gt_one:
    /* asin(x > 1) */
    /* asind(x > 1) */
    {
       ERRNO_DOMAIN; break;
    }
  case asinf_gt_one:
  case asindf_gt_one:
    /* asinf(x > 1) */
    /* asindf(x > 1) */
    {
       ERRNO_DOMAIN; break;
    }
  case remainderl_by_zero:
  case fmodl_by_zero:
    /* fmodl(x,0) */
    {
       ERRNO_DOMAIN; break;
    }
  case remainder_by_zero:
  case fmod_by_zero:
    /* fmod(x,0) */
    {
       ERRNO_DOMAIN; break;
    }
  case remainderf_by_zero:
  case fmodf_by_zero:
    /* fmodf(x,0) */
    {
       ERRNO_DOMAIN; break;
    }
  case coshl_overflow:
    /* coshl overflows */
    {
       RETVAL_HUGE_VALL; ERRNO_RANGE; break;
    }
  case cosh_overflow:
    /* cosh overflows */
    {
       RETVAL_HUGE_VALD; ERRNO_RANGE; break;
    }
  case coshf_overflow:
    /* coshf overflows */
    {
       RETVAL_HUGE_VALF; ERRNO_RANGE; break;
    }
  case sinhl_overflow:
    /* sinhl overflows */
    {
       if (INPUT_XL > ZEROL_VALUE /*0*/) RETVAL_HUGE_VALL;
       else RETVAL_NEG_HUGE_VALL;
       ERRNO_RANGE; break;
    }
  case sinh_overflow:
    /* sinh overflows */
    {
       if (INPUT_XD > ZEROD_VALUE /*0*/) RETVAL_HUGE_VALD;
       else RETVAL_NEG_HUGE_VALD;
       ERRNO_RANGE; break;
    }
  case sinhf_overflow:
    /* sinhf overflows */
    {
       if (INPUT_XF > ZEROF_VALUE /*0*/) RETVAL_HUGE_VALF;
       else RETVAL_NEG_HUGE_VALF;
       ERRNO_RANGE; break;
    }
  case logbl_zero:
   /* logbl(0) */
   {
      ERRNO_DOMAIN; break;
   }
  case logb_zero:
   /* logb(0) */
   {
      ERRNO_DOMAIN; break;
   }
  case logbf_zero:
   /* logbf(0) */
   {
      ERRNO_DOMAIN; break;
   }
  case ilogbl_zero:
   /* ilogbl(0) */
   {
      ERRNO_RANGE; break;
   }
  case ilogb_zero:
   /* ilogb(0) */
   {
      ERRNO_RANGE; break;
   }
  case ilogbf_zero:
   /* ilogbf(0) */
   {
      ERRNO_RANGE; break;
   }
  default:
    break;
}
return;
/* _POSIX_ */
}

/*******************************/
/* __SVID__ and __XOPEN__ Path */
/*******************************/
else
{
  switch(input_tag)
  {
  case ldexpl_overflow:
  case ldexpl_underflow:
  case ldexp_overflow:
  case ldexp_underflow:
  case ldexpf_overflow:
  case ldexpf_underflow:
  case scalbnl_overflow:
  case scalbnl_underflow:
  case scalbn_overflow:
  case scalbn_underflow:
  case scalbnf_overflow:
  case scalbnf_underflow:
  case scalblnl_overflow:
  case scalblnl_underflow:
  case scalbln_overflow:
  case scalbln_underflow:
  case scalblnf_overflow:
  case scalblnf_underflow:
  case tandl_overflow:
  case tand_overflow:
  case tandf_overflow:
  case cotdl_overflow:
  case cotd_overflow:
  case cotdf_overflow:
  case cotl_overflow:
  case cot_overflow:
  case cotf_overflow:
  case annuityl_overflow:
  case annuityl_underflow:
  case annuity_overflow:
  case annuity_underflow:
  case annuityf_overflow:
  case annuityf_underflow:
  case compoundl_overflow:
  case compoundl_underflow:
  case compound_overflow:
  case compound_underflow:
  case compoundf_overflow:
  case compoundf_underflow:
  {
       ERRNO_RANGE; break;
  }
  case annuityl_by_zero:
  case annuityl_less_m1:
  case annuity_by_zero:
  case annuity_less_m1:
  case annuityf_by_zero:
  case annuityf_less_m1:
  case compoundl_by_zero:
  case compoundl_less_m1:
  case compound_by_zero:
  case compound_less_m1:
  case compoundf_by_zero:
  case compoundf_less_m1:
  {
       ERRNO_DOMAIN; break;
  }
  case sqrtl_negative:
    /* sqrtl(x < 0) */
    {
       DOMAINL; NAMEL = (char *) "sqrtl";
       ifSVID
       {
          RETVAL_ZEROL;
          NOT_MATHERRL
          {
            WRITEL_SQRT;
            ERRNO_DOMAIN;
          }
       }
       else
       { /* NaN already computed */
          NOT_MATHERRL {ERRNO_DOMAIN;}
       }
       *(long double *)retval = excl.retval;
       break;
    }
  case sqrt_negative:
    /* sqrt(x < 0) */
    {
       DOMAIND; NAMED = (char *) "sqrt";
       ifSVID
       {

         RETVAL_ZEROD;
         NOT_MATHERRD
         {
           WRITED_SQRT;
           ERRNO_DOMAIN;
         }
       }
       else
       { /* NaN already computed */
         NOT_MATHERRD {ERRNO_DOMAIN;}
       }
       *(double *)retval = exc.retval;
       break;
    }
  case sqrtf_negative:
    /* sqrtf(x < 0) */
    {
       DOMAINF; NAMEF = (char *) "sqrtf";
       ifSVID
       {
         RETVAL_ZEROF;
         NOT_MATHERRF
         {
           WRITEF_SQRT;
           ERRNO_DOMAIN;
         }
       }
       else
       {
         NOT_MATHERRF {ERRNO_DOMAIN;}
       }
       *(float *)retval = excf.retval;
       break;
    }
  case logl_zero:
    /* logl(0) */
    {
       SINGL; NAMEL = (char *) "logl";
       ifSVID
       {
         RETVAL_NEG_HUGEL;
         NOT_MATHERRL
         {
           WRITEL_LOG_ZERO;
           ERRNO_DOMAIN;
         }
       }
       else
       {
         RETVAL_NEG_HUGE_VALL;
         NOT_MATHERRL {ERRNO_DOMAIN;}
       }
       *(long double *)retval = excl.retval;
       break;
    }
  case log_zero:
    /* log(0) */
    {
       SINGD; NAMED = (char *) "log";
       ifSVID
       {
         RETVAL_NEG_HUGED;
         NOT_MATHERRD
         {
           WRITED_LOG_ZERO;
           ERRNO_DOMAIN;
         }
       }
       else
       {
         RETVAL_NEG_HUGE_VALD;
         NOT_MATHERRD {ERRNO_DOMAIN;}
       }
       *(double *)retval = exc.retval;
       break;
    }
  case logf_zero:
    /* logf(0) */
    {
       SINGF; NAMEF = (char *) "logf";
       ifSVID
       {
         RETVAL_NEG_HUGEF;
         NOT_MATHERRF
         {
            WRITEF_LOG_ZERO;
            ERRNO_DOMAIN;
         }
       }
       else
       {
         RETVAL_NEG_HUGE_VALF;
         NOT_MATHERRF {ERRNO_DOMAIN;}
       }
       *(float *)retval = excf.retval;
       break;
    }

  case logl_negative:
    /* logl(x < 0) */
    {
       DOMAINL; NAMEL = (char *) "logl";
       ifSVID
       {
         RETVAL_NEG_HUGEL;
         NOT_MATHERRL
         {
           WRITEL_LOG_NEGATIVE;
           ERRNO_DOMAIN;
         }
       }
       else
       {
         RETVAL_NEG_HUGE_VALL;
         NOT_MATHERRL {ERRNO_DOMAIN;}
       }
       *(long double *)retval = excl.retval;
       break;
    }
  case log_negative:
    /* log(x < 0) */
    {
       DOMAIND; NAMED = (char *) "log";
       ifSVID
       {
         RETVAL_NEG_HUGED;
         NOT_MATHERRD
         {
           WRITED_LOG_NEGATIVE;
           ERRNO_DOMAIN;
         }
       }
       else
       {
         RETVAL_NEG_HUGE_VALD;
         NOT_MATHERRD {ERRNO_DOMAIN;}
       }
       *(double *)retval = exc.retval;
       break;
    }
  case logf_negative:
    /* logf(x < 0) */
    {
       DOMAINF; NAMEF = (char *) "logf";
       ifSVID
       {
         RETVAL_NEG_HUGEF;
         NOT_MATHERRF
         {
           WRITEF_LOG_NEGATIVE;
           ERRNO_DOMAIN;
         }
       }
       else
       {
         RETVAL_NEG_HUGE_VALF;
         NOT_MATHERRF{ERRNO_DOMAIN;}
       }
       *(float *)retval = excf.retval;
       break;
    }
  case log1pl_zero:
    /* log1pl(-1) */
    {
       SINGL; NAMEL = (char *) "log1pl";
       ifSVID
       {
         RETVAL_NEG_HUGEL;
         NOT_MATHERRL
         {
           WRITEL_LOG1P_ZERO;
           ERRNO_DOMAIN;
         }
       }
       else
       {
         RETVAL_NEG_HUGE_VALL;
         NOT_MATHERRL {ERRNO_DOMAIN;}
       }
       *(long double *)retval = excl.retval;
       break;
    }
  case log1p_zero:
    /* log1p(-1) */
    {
       SINGD; NAMED = (char *) "log1p";
       ifSVID
       {
         RETVAL_NEG_HUGED;
         NOT_MATHERRD
         {
           WRITED_LOG1P_ZERO;
           ERRNO_DOMAIN;
         }
       }
       else
       {
         RETVAL_NEG_HUGE_VALD;
         NOT_MATHERRD {ERRNO_DOMAIN;}
       }
       *(double *)retval = exc.retval;
       break;
    }
  case log1pf_zero:
    /* log1pf(-1) */
    {
       SINGF; NAMEF = (char *) "log1pf";
       ifSVID
       {
         RETVAL_NEG_HUGEF;
         NOT_MATHERRF
         {
           WRITEF_LOG1P_ZERO;
           ERRNO_DOMAIN;
         }
       }
       else
       {
         RETVAL_NEG_HUGE_VALF;
         NOT_MATHERRF {ERRNO_DOMAIN;}
       }
       *(float *)retval = excf.retval;
       break;
    }
 case log1pl_negative:
   /* log1pl(x < -1) */
   {
      DOMAINL; NAMEL = (char *) "log1pl";
      ifSVID
      {
        RETVAL_NEG_HUGEL;
        NOT_MATHERRL
        {
          WRITEL_LOG1P_NEGATIVE;
          ERRNO_DOMAIN;
        }
      }
      else
      {
        RETVAL_NEG_HUGE_VALL;
        NOT_MATHERRL {ERRNO_DOMAIN;}
      }
      *(long double *)retval = excl.retval;
      break;
   }
 case log1p_negative:
   /* log1p(x < -1) */
   {
      DOMAIND; NAMED = (char *) "log1p";
      ifSVID
      {
        RETVAL_NEG_HUGED;
        NOT_MATHERRD
        {
          WRITED_LOG1P_NEGATIVE;
          ERRNO_DOMAIN;
        }
      }
      else
      {
        RETVAL_NEG_HUGE_VALD;
        NOT_MATHERRD {ERRNO_DOMAIN;}
      }
      *(double *)retval = exc.retval;
      break;
   }
 case log1pf_negative:
   /* log1pf(x < -1) */
   {
      DOMAINF; NAMEF = (char *) "log1pf";
      ifSVID
      {
        RETVAL_NEG_HUGEF;
        NOT_MATHERRF
        {
          WRITEF_LOG1P_NEGATIVE;
          ERRNO_DOMAIN;
        }
      }
      else
      {
        RETVAL_NEG_HUGE_VALF;
        NOT_MATHERRF {ERRNO_DOMAIN;}
      }
      *(float *)retval = excf.retval;
      break;
   }
  case log10l_zero:
    /* log10l(0) */
    {
       SINGL; NAMEL = (char *) "log10l";
       ifSVID
       {
         RETVAL_NEG_HUGEL;
         NOT_MATHERRL
         {
            WRITEL_LOG10_ZERO;
            ERRNO_DOMAIN;
         }
       }
       else
       {
         RETVAL_NEG_HUGE_VALL;
         NOT_MATHERRL {ERRNO_DOMAIN;}
       }
       *(long double *)retval = excl.retval;
       break;
    }
  case log10_zero:
    /* log10(0) */
    {
       SINGD; NAMED = (char *) "log10";
       ifSVID
       {
         RETVAL_NEG_HUGED;
         NOT_MATHERRD
         {
           WRITED_LOG10_ZERO;
           ERRNO_DOMAIN;
         }
       }
       else
       {
         RETVAL_NEG_HUGE_VALD;
         NOT_MATHERRD {ERRNO_DOMAIN;}
       }
       *(double *)retval = exc.retval;
       break;
    }
  case log10f_zero:
    /* log10f(0) */
    {
       SINGF; NAMEF = (char *) "log10f";
       ifSVID
       {
         RETVAL_NEG_HUGEF;
         NOT_MATHERRF
         {
          WRITEF_LOG10_ZERO;
          ERRNO_DOMAIN;
         }
       }
       else
       {
         RETVAL_NEG_HUGE_VALF;
         NOT_MATHERRF {ERRNO_DOMAIN;}
       }
       *(float *)retval = excf.retval;
       break;
    }
  case log10l_negative:
    /* log10l(x < 0) */
    {
       DOMAINL; NAMEL = (char *) "log10l";
       ifSVID
       {
         RETVAL_NEG_HUGEL;
         NOT_MATHERRL
         {
           WRITEL_LOG10_NEGATIVE;
           ERRNO_DOMAIN;
         }
       }
       else
       {
         RETVAL_NEG_HUGE_VALL;
         NOT_MATHERRL {ERRNO_DOMAIN;}
       }
       *(long double *)retval = excl.retval;
       break;
    }
  case log10_negative:
    /* log10(x < 0) */
    {
       DOMAIND; NAMED = (char *) "log10";
       ifSVID
       {
         RETVAL_NEG_HUGED;
         NOT_MATHERRD
         {
           WRITED_LOG10_NEGATIVE;
           ERRNO_DOMAIN;
         }
       }
       else
       {
         RETVAL_NEG_HUGE_VALD;
         NOT_MATHERRD {ERRNO_DOMAIN;}
       }
       *(double *)retval = exc.retval;
       break;
    }
  case log10f_negative:
    /* log10f(x < 0) */
    {
       DOMAINF; NAMEF = (char *) "log10f";
       ifSVID
       {
         RETVAL_NEG_HUGEF;
         NOT_MATHERRF
         {
           WRITEF_LOG10_NEGATIVE;
           ERRNO_DOMAIN;
         }
       }
       else
       {
         RETVAL_NEG_HUGE_VALF;
         NOT_MATHERRF {ERRNO_DOMAIN;}
       }
       *(float *)retval = excf.retval;
       break;
    }
  case log2l_zero:
    /* log2l(0) */
    {
       SINGL; NAMEL = (char *) "log2l";
       ifSVID
       {
         RETVAL_NEG_HUGEL;
         NOT_MATHERRL
         {
           WRITEL_LOG2_ZERO;
           ERRNO_DOMAIN;
         }
       }
       else
       {
         RETVAL_NEG_HUGE_VALL;
         NOT_MATHERRL {ERRNO_DOMAIN;}
       }
       *(long double *)retval = excl.retval;
       break;
    }
  case log2_zero:
    /* log2(0) */
    {
       SINGD; NAMED = (char *) "log2";
       ifSVID
       {
         RETVAL_NEG_HUGED;
         NOT_MATHERRD
         {
           WRITED_LOG2_ZERO;
           ERRNO_DOMAIN;
         }
       }
       else
       {
         RETVAL_NEG_HUGE_VALD;
         NOT_MATHERRD {ERRNO_DOMAIN;}
       }
       *(double *)retval = exc.retval;
       break;
    }
  case log2f_zero:
    /* log2f(0) */
    {
       SINGF; NAMEF = (char *) "log2f";
       ifSVID
       {
         RETVAL_NEG_HUGEF;
         NOT_MATHERRF
         {
          WRITEF_LOG2_ZERO;
          ERRNO_DOMAIN;
         }
       }
       else
       {
         RETVAL_NEG_HUGE_VALF;
         NOT_MATHERRF {ERRNO_DOMAIN;}
       }
       *(float *)retval = excf.retval;
       break;
    }
  case log2l_negative:
    /* log2l(x < 0) */
    {
       DOMAINL; NAMEL = (char *) "log2l";
       ifSVID
       {
         RETVAL_NEG_HUGEL;
         NOT_MATHERRL
         {
           WRITEL_LOG2_NEGATIVE;
           ERRNO_DOMAIN;
         }
       }
       else
       {
         RETVAL_NEG_HUGE_VALL;
         NOT_MATHERRL {ERRNO_DOMAIN;}
       }
       *(long double *)retval = excl.retval;
       break;
    }
  case log2_negative:
    /* log2(x < 0) */
    {
       DOMAIND; NAMED = (char *) "log2";
       ifSVID
       {
         RETVAL_NEG_HUGED;
         NOT_MATHERRD
         {
           WRITED_LOG2_NEGATIVE;
           ERRNO_DOMAIN;
         }
       }
       else
       {
         RETVAL_NEG_HUGE_VALD;
         NOT_MATHERRD {ERRNO_DOMAIN;}
       }
       *(double *)retval = exc.retval;
       break;
    }
  case log2f_negative:
    /* log2f(x < 0) */
    {
       DOMAINF; NAMEF = (char *) "log2f";
       ifSVID
       {
         RETVAL_NEG_HUGEF;
         NOT_MATHERRF
         {
           WRITEF_LOG2_NEGATIVE;
           ERRNO_DOMAIN;
         }
       }
       else
       {
         RETVAL_NEG_HUGE_VALF;
         NOT_MATHERRF {ERRNO_DOMAIN;}
       }
       *(float *)retval = excf.retval;
       break;
    }
  case expl_overflow:
    /* expl overflow */
    {
       OVERFLOWL; NAMEL = (char *) "expl";
       ifSVID
       {
         RETVAL_HUGEL;
       }
       else
       {
       RETVAL_HUGE_VALL;
       }
       NOT_MATHERRL {ERRNO_RANGE;}
       *(long double *)retval = excl.retval;
       break;
    }
  case exp_overflow:
    /* exp overflow */
    {
       OVERFLOWD; NAMED = (char *) "exp";
       ifSVID
       {
         RETVAL_HUGED;
       }
       else
       {
         RETVAL_HUGE_VALD;
       }
       NOT_MATHERRD {ERRNO_RANGE;}
       *(double *)retval = exc.retval;
       break;
    }
  case expf_overflow:
    /* expf overflow */
    {
       OVERFLOWF; NAMEF = (char *) "expf";
       ifSVID
       {
         RETVAL_HUGEF;
       }
       else
       {
         RETVAL_HUGE_VALF;
       }
       NOT_MATHERRF {ERRNO_RANGE;}
       *(float *)retval = excf.retval;
       break;
    }
  case expl_underflow:
    /* expl underflow */
    {
       UNDERFLOWL; NAMEL = (char *) "expl"; RETVAL_ZEROL;
       NOT_MATHERRL {ERRNO_RANGE;}
       *(long double *)retval = excl.retval;
       break;
    }
  case exp_underflow:
    /* exp underflow */
    {
       UNDERFLOWD; NAMED = (char *) "exp"; RETVAL_ZEROD;
       NOT_MATHERRD {ERRNO_RANGE;}
       *(double *)retval = exc.retval;
       break;
    }
  case expf_underflow:
    /* expf underflow */
    {
       UNDERFLOWF; NAMEF = (char *) "expf"; RETVAL_ZEROF;
       NOT_MATHERRF {ERRNO_RANGE;}
       *(float *)retval = excf.retval;
       break;
    }
  case powl_zero_to_zero:
    /* powl 0**0 */
    {
       DOMAINL; NAMEL = (char *) "powl";
       ifSVID
       {
         RETVAL_ZEROL;
         NOT_MATHERRL
         {
            WRITEL_POW_ZERO_TO_ZERO;
            ERRNO_DOMAIN;
         }
         *(long double *)retval = excl.retval;
       }
       else RETVAL_ONEL;
       break;
    }
  case pow_zero_to_zero:
    /* pow 0**0 */
    {
       DOMAIND; NAMED = (char *) "pow";
       ifSVID
       {
         RETVAL_ZEROD;
         NOT_MATHERRD
         {
            WRITED_POW_ZERO_TO_ZERO;
            ERRNO_DOMAIN;
         }
         *(double *)retval = exc.retval;
       }
       else RETVAL_ONED;
       break;
    }
  case powf_zero_to_zero:
    /* powf 0**0 */
    {
       DOMAINF; NAMEF = (char *) "powf";
       ifSVID
       {
         RETVAL_ZEROF;
         NOT_MATHERRF
         {
          WRITEF_POW_ZERO_TO_ZERO;
          ERRNO_DOMAIN;
         }
         *(float *)retval = excf.retval;
       }
       else RETVAL_ONEF;
       break;
    }
  case powl_overflow:
    /* powl(x,y) overflow */
    {
       OVERFLOWL; NAMEL = (char *) "powl";
       ifSVID
       {
         if (INPUT_RESL < ZEROL_VALUE /*0*/) RETVAL_NEG_HUGEL;
         else RETVAL_HUGEL;
       }
       else
       {
         if (INPUT_RESL < ZEROL_VALUE /*0*/) RETVAL_NEG_HUGE_VALL;
         else RETVAL_HUGE_VALL;
       }
       NOT_MATHERRL {ERRNO_RANGE;}
       *(long double *)retval = excl.retval;
       break;
    }
  case pow_overflow:
    /* pow(x,y) overflow */
    {
       OVERFLOWD; NAMED = (char *) "pow";
       ifSVID
       {
         if (INPUT_RESD < ZEROD_VALUE /*0*/) RETVAL_NEG_HUGED;
         else RETVAL_HUGED;
       }
       else
       {
         if (INPUT_RESD < ZEROD_VALUE /*0*/) RETVAL_NEG_HUGE_VALD;
         else RETVAL_HUGE_VALD;
       }
       NOT_MATHERRD {ERRNO_RANGE;}
       *(double *)retval = exc.retval;
       break;
    }
  case powf_overflow:
    /* powf(x,y) overflow */
    {
       OVERFLOWF; NAMEF = (char *) "powf";
       ifSVID
       {
         if (INPUT_RESF < ZEROF_VALUE /*0*/) RETVAL_NEG_HUGEF;
         else RETVAL_HUGEF;
       }
       else
       {
         if (INPUT_RESF < ZEROF_VALUE /*0*/) RETVAL_NEG_HUGE_VALF;
         else RETVAL_HUGE_VALF;
       }
       NOT_MATHERRF {ERRNO_RANGE;}
       *(float *)retval = excf.retval;
       break;
    }
  case powl_underflow:
    /* powl(x,y) underflow */
    {
       UNDERFLOWL; NAMEL = (char *) "powl"; RETVAL_ZEROL;
       NOT_MATHERRL {ERRNO_RANGE;}
       *(long double *)retval = excl.retval;
       break;
    }
  case pow_underflow:
    /* pow(x,y) underflow */
    {
       UNDERFLOWD; NAMED = (char *) "pow"; RETVAL_ZEROD;
       NOT_MATHERRD {ERRNO_RANGE;}
       *(double *)retval = exc.retval;
       break;
    }
  case powf_underflow:
    /* powf(x,y) underflow */
    {
       UNDERFLOWF; NAMEF = (char *) "powf"; RETVAL_ZEROF;
       NOT_MATHERRF {ERRNO_RANGE;}
       *(float *)retval = excf.retval;
       break;
    }
  case powl_zero_to_negative:
    /* 0 to neg */
    {
       DOMAINL; NAMEL = (char *) "powl";
       ifSVID
       {
         RETVAL_ZEROL;
         NOT_MATHERRL
         {
           WRITEL_POW_ZERO_TO_NEGATIVE;
           ERRNO_DOMAIN;
         }
       }
       else
       {
         RETVAL_NEG_HUGE_VALL;
         NOT_MATHERRL {ERRNO_DOMAIN;}
       }
       *(long double *)retval = excl.retval;
       break;
    }
  case pow_zero_to_negative:
    /* 0**neg */
    {
       DOMAIND; NAMED = (char *) "pow";
       ifSVID
       {
         RETVAL_ZEROD;
         NOT_MATHERRD
         {
           WRITED_POW_ZERO_TO_NEGATIVE;
           ERRNO_DOMAIN;
         }
       }
       else
       {
         RETVAL_NEG_HUGE_VALD;
         NOT_MATHERRD {ERRNO_DOMAIN;}
       }
       *(double *)retval = exc.retval;
       break;
    }
  case powf_zero_to_negative:
    /* 0**neg */
    {
       DOMAINF; NAMEF = (char *) "powf";
       ifSVID
       {
         RETVAL_ZEROF;
         NOT_MATHERRF
         {
            WRITEF_POW_ZERO_TO_NEGATIVE;
            ERRNO_DOMAIN;
         }
       }
       else
       {
         RETVAL_NEG_HUGE_VALF;
         NOT_MATHERRF {ERRNO_DOMAIN;}
       }
       *(float *)retval = excf.retval;
       break;
    }
  case powl_neg_to_non_integer:
    /* neg**non_integral */
    {
       DOMAINL; NAMEL = (char *) "powl";
       ifSVID
       {
         RETVAL_ZEROL;
         NOT_MATHERRL
         {
           WRITEL_POW_NEG_TO_NON_INTEGER;
           ERRNO_DOMAIN;
         }
       }
       else
       {
         NOT_MATHERRL {ERRNO_DOMAIN;}
       }
       *(long double *)retval = excl.retval;
       break;
    }
  case pow_neg_to_non_integer:
    /* neg**non_integral */
    {
       DOMAIND; NAMED = (char *) "pow";
       ifSVID
       {
         RETVAL_ZEROD;
         NOT_MATHERRD
         {
            WRITED_POW_NEG_TO_NON_INTEGER;
            ERRNO_DOMAIN;
         }
       }
       else
       {
         NOT_MATHERRD {ERRNO_DOMAIN;}
       }
       *(double *)retval = exc.retval;
       break;
    }
  case powf_neg_to_non_integer:
    /* neg**non-integral */
    {
       DOMAINF; NAMEF = (char *) "powf";
       ifSVID
       {
         RETVAL_ZEROF;
         NOT_MATHERRF
         {
            WRITEF_POW_NEG_TO_NON_INTEGER;
            ERRNO_DOMAIN;
         }
       }
       else
       {
         NOT_MATHERRF {ERRNO_DOMAIN;}
       }
       *(float *)retval = excf.retval;
       break;
    }
  case powl_nan_to_zero:
    /* pow(NaN,0.0) */
    /* Special Error */
    {
       DOMAINL; NAMEL = (char *) "powl";
       *(long double *)retval = *(long double *)arg1;
       NOT_MATHERRL {ERRNO_DOMAIN;}
       *(long double *)retval = excl.retval;
       break;
    }
  case pow_nan_to_zero:
    /* pow(NaN,0.0) */
    /* Special Error */
    {
       DOMAIND; NAMED = (char *) "pow";
       *(double *)retval = *(double *)arg1;
       NOT_MATHERRD {ERRNO_DOMAIN;}
       *(double *)retval = exc.retval;
       break;
    }
  case powf_nan_to_zero:
    /* powf(NaN,0.0) */
    /* Special Error */
    {
       DOMAINF; NAMEF = (char *) "powf";
       *(float *)retval = *(float *)arg1;
       NOT_MATHERRF {ERRNO_DOMAIN;}
       *(float *)retval = excf.retval;
       break;
    }
  case atan2l_zero:
    /* atan2l(0.0,0.0) */
    {
       DOMAINL; NAMEL = (char *) "atan2l";
       RETVAL_ZEROL;
       NOT_MATHERRL
       {
         ifSVID
         {
            WRITEL_ATAN2_ZERO_BY_ZERO;
         }
         ERRNO_DOMAIN;
       }
       *(long double *)retval = excl.retval;
       break;
    }
  case atan2_zero:
    /* atan2(0.0,0.0) */
    {
       DOMAIND; NAMED = (char *) "atan2";
       RETVAL_ZEROD;
       NOT_MATHERRD
       {
         ifSVID
         {
            WRITED_ATAN2_ZERO_BY_ZERO;
         }
         ERRNO_DOMAIN;
       }
       *(double *)retval = exc.retval;
       break;
    }
  case atan2f_zero:
    /* atan2f(0.0,0.0) */
    {
       DOMAINF; NAMEF = (char *) "atan2f";
       RETVAL_ZEROF;
       NOT_MATHERRF
       {
         ifSVID
         {
            WRITEF_ATAN2_ZERO_BY_ZERO;
         }
         ERRNO_DOMAIN;
       }
       *(float *)retval = excf.retval;
       break;
    }
  case atan2dl_zero:
    /* atan2dl(0.0,0.0) */
    {
       DOMAINL; NAMEL = (char *) "atan2dl";
       RETVAL_ZEROL;
       NOT_MATHERRL
       {
         ifSVID
         {
            WRITEL_ATAN2D_ZERO_BY_ZERO;
         }
         ERRNO_DOMAIN;
       }
       *(long double *)retval = excl.retval;
       break;
    }
  case atan2d_zero:
    /* atan2d(0.0,0.0) */
    {
       DOMAIND; NAMED = (char *) "atan2d";
       RETVAL_ZEROD;
       NOT_MATHERRD
       {
         ifSVID
         {
            WRITED_ATAN2D_ZERO_BY_ZERO;
         }
         ERRNO_DOMAIN;
       }
       *(double *)retval = exc.retval;
       break;
    }
  case atan2df_zero:
    /* atan2df(0.0,0.0) */
    {
       DOMAINF; NAMEF = (char *) "atan2df";
       RETVAL_ZEROF;
       NOT_MATHERRF
       {
         ifSVID
         {
            WRITEF_ATAN2D_ZERO_BY_ZERO;
         }
         ERRNO_DOMAIN;
       }
       *(float *)retval = excf.retval;
       break;
    }
  case expm1_overflow:
    /* expm1(finite) overflow */
    /* Overflow is the only documented */
    /* special value. */
    {
      ERRNO_RANGE;
      break;
    }
  case expm1f_overflow:
    /* expm1f(finite) overflow */
    {
      ERRNO_RANGE;
      break;
    }
  case expm1_underflow:
    /* expm1(finite) underflow */
    /* Underflow is not documented */
    /* special value. */
    {
      ERRNO_RANGE;
      break;
    }
  case expm1f_underflow:
    /* expm1f(finite) underflow */
    {
      ERRNO_RANGE;
      break;
    }
  case scalbl_underflow:
    /* scalbl underflow */
    {
       UNDERFLOWL; NAMEL = (char *) "scalbl";
       if (INPUT_XL < ZEROL_VALUE /*0.0L*/) RETVAL_NEG_ZEROL;
       else  RETVAL_ZEROL;
       NOT_MATHERRL {ERRNO_RANGE;}
       *(long double *)retval = excl.retval;
       break;
    }
  case scalb_underflow:
    /* scalb underflow */
    {
       UNDERFLOWD; NAMED = (char *) "scalb";
       if (INPUT_XD < ZEROD_VALUE /*0.0*/) RETVAL_NEG_ZEROD;
       else  RETVAL_ZEROD;
       NOT_MATHERRD {ERRNO_RANGE;}
       *(double *)retval = exc.retval;
       break;
    }
  case scalbf_underflow:
    /* scalbf underflow */
    {
       UNDERFLOWF; NAMEF = (char *) "scalbf";
       if (INPUT_XF < ZEROF_VALUE /*0.0*/) RETVAL_NEG_ZEROF;
       else  RETVAL_ZEROF;
       NOT_MATHERRF {ERRNO_RANGE;}
       *(float *)retval = excf.retval;
       break;
    }
  case scalbl_overflow:
    /* scalbl overflow */
    {
       OVERFLOWL; NAMEL = (char *) "scalbl";
       if (INPUT_XL < ZEROL_VALUE /*0*/) RETVAL_NEG_HUGE_VALL;
       else RETVAL_HUGE_VALL;
       NOT_MATHERRL {ERRNO_RANGE;}
       *(long double *)retval = excl.retval;
       break;
    }
  case scalb_overflow:
    /* scalb overflow */
    {
       OVERFLOWD; NAMED = (char *) "scalb";
       if (INPUT_XD < ZEROD_VALUE /*0*/) RETVAL_NEG_HUGE_VALD;
       else RETVAL_HUGE_VALD;
       NOT_MATHERRD {ERRNO_RANGE;}
       *(double *)retval = exc.retval;
       break;
    }
  case scalbf_overflow:
    /* scalbf overflow */
    {
       OVERFLOWF; NAMEF = (char *) "scalbf";
       if (INPUT_XF < ZEROF_VALUE /*0*/) RETVAL_NEG_HUGE_VALF;
       else RETVAL_HUGE_VALF;
       NOT_MATHERRF {ERRNO_RANGE;}
       *(float *)retval = excf.retval;
       break;
    }
  case hypotl_overflow:
    /* hypotl overflow */
    {
       OVERFLOWL; NAMEL = (char *) "hypotl";
       ifSVID
       {
         RETVAL_HUGEL;
       }
       else
       {
         RETVAL_HUGE_VALL;
       }
       NOT_MATHERRL {ERRNO_RANGE;}
       *(long double *)retval = excl.retval;
       break;
    }
  case hypot_overflow:
    /* hypot overflow */
    {
       OVERFLOWD; NAMED = (char *) "hypot";
       ifSVID
       {
         RETVAL_HUGED;
       }
       else
       {
         RETVAL_HUGE_VALD;
       }
       NOT_MATHERRD {ERRNO_RANGE;}
       *(double *)retval = exc.retval;
       break;
    }
  case hypotf_overflow:
    /* hypotf overflow */
    {
       OVERFLOWF; NAMEF = (char *) "hypotf";
       ifSVID
       {
         RETVAL_HUGEF;
       }
       else
       {
         RETVAL_HUGE_VALF;
       }
       NOT_MATHERRF {ERRNO_RANGE;}
       *(float *)retval = excf.retval;
       break;
    }
  case acosl_gt_one:
    /* acosl(x > 1) */
    {
       DOMAINL; NAMEL = (char *) "acosl";
       RETVAL_ZEROL;
       ifSVID
       {
         NOT_MATHERRL
         {
           WRITEL_ACOS;
           ERRNO_DOMAIN;
         }
       }
       else
       {
         NOT_MATHERRL {ERRNO_DOMAIN;}
       }
       *(long double *)retval = excl.retval;
       break;
    }
  case acos_gt_one:
    /* acos(x > 1) */
    {
       DOMAIND; NAMED = (char *) "acos";
       RETVAL_ZEROD;
       ifSVID
       {
         NOT_MATHERRD
         {
           WRITED_ACOS;
           ERRNO_DOMAIN;
         }
       }
       else
       {
         NOT_MATHERRD {ERRNO_DOMAIN;}
       }
       *(double *)retval = exc.retval;
       break;
    }
  case acosf_gt_one:
    /* acosf(x > 1) */
    {
       DOMAINF; NAMEF = (char *) "acosf";
       RETVAL_ZEROF;
       ifSVID
       {
         NOT_MATHERRF
         {
           WRITEF_ACOS;
           ERRNO_DOMAIN;
         }
       }
       else
       {
         NOT_MATHERRF {ERRNO_DOMAIN;}
       }
       *(float *)retval = excf.retval;
       break;
    }
  case asinl_gt_one:
    /* asinl(x > 1) */
    {
       DOMAINL; NAMEL = (char *) "asinl";
       RETVAL_ZEROL;
       ifSVID
       {
         NOT_MATHERRL
         {
           WRITEL_ASIN;
           ERRNO_DOMAIN;
         }
       }
       else
       {
         NOT_MATHERRL {ERRNO_DOMAIN;}
       }
       *(long double *)retval = excl.retval;
       break;
    }
  case asin_gt_one:
    /* asin(x > 1) */
    {
       DOMAIND; NAMED = (char *) "asin";
       RETVAL_ZEROD;
       ifSVID
       {
         NOT_MATHERRD
         {
           WRITED_ASIN;
           ERRNO_DOMAIN;
         }
       }
       else
       {
         NOT_MATHERRD {ERRNO_DOMAIN;}
       }
       *(double *)retval = exc.retval;
       break;
    }
  case asinf_gt_one:
    /* asinf(x > 1) */
    {
       DOMAINF; NAMEF = (char *) "asinf";
       RETVAL_ZEROF;
       ifSVID
       {
         NOT_MATHERRF
         {
            WRITEF_ASIN;
            ERRNO_DOMAIN;
         }
       }
       else
       {
         NOT_MATHERRF {ERRNO_DOMAIN;}
       }
       *(float *)retval = excf.retval;
       break;
    }
  case acosdl_gt_one:
    /* acosdl(x > 1) */
    {
       DOMAINL; NAMEL = (char *) "acosdl";
       RETVAL_ZEROL;
       ifSVID
       {
         NOT_MATHERRL
         {
           WRITEL_ACOSD;
           ERRNO_DOMAIN;
         }
       }
       else
       {
         NOT_MATHERRL {ERRNO_DOMAIN;}
       }
       *(long double *)retval = excl.retval;
       break;
    }
  case acosd_gt_one:
    /* acosd(x > 1) */
    {
       DOMAIND; NAMED = (char *) "acosd";
       RETVAL_ZEROD;
       ifSVID
       {
         NOT_MATHERRD
         {
           WRITED_ACOSD;
           ERRNO_DOMAIN;
         }
       }
       else
       {
         NOT_MATHERRD {ERRNO_DOMAIN;}
       }
       *(double *)retval = exc.retval;
       break;
    }
  case acosdf_gt_one:
    /* acosdf(x > 1) */
    {
       DOMAINF; NAMEF = (char *) "acosdf";
       RETVAL_ZEROF;
       ifSVID
       {
         NOT_MATHERRF
         {
           WRITEF_ACOSD;
           ERRNO_DOMAIN;
         }
       }
       else
       {
         NOT_MATHERRF {ERRNO_DOMAIN;}
       }
       *(float *)retval = excf.retval;
       break;
    }
  case asindl_gt_one:
    /* asindl(x > 1) */
    {
       DOMAINL; NAMEL = (char *) "asindl";
       RETVAL_ZEROL;
       ifSVID
       {
         NOT_MATHERRL
         {
           WRITEL_ASIND;
           ERRNO_DOMAIN;
         }
       }
       else
       {
         NOT_MATHERRL {ERRNO_DOMAIN;}
       }
       *(long double *)retval = excl.retval;
       break;
    }
  case asind_gt_one:
    /* asind(x > 1) */
    {
       DOMAIND; NAMED = (char *) "asind";
       RETVAL_ZEROD;
       ifSVID
       {
         NOT_MATHERRD
         {
           WRITED_ASIND;
           ERRNO_DOMAIN;
         }
       }
       else
       {
         NOT_MATHERRD {ERRNO_DOMAIN;}
       }
       *(double *)retval = exc.retval;
       break;
    }
  case asindf_gt_one:
    /* asindf(x > 1) */
    {
       DOMAINF; NAMEF = (char *) "asindf";
       RETVAL_ZEROF;
       ifSVID
       {
         NOT_MATHERRF
         {
            WRITEF_ASIND;
            ERRNO_DOMAIN;
         }
       }
       else
       {
         NOT_MATHERRF {ERRNO_DOMAIN;}
       }
       *(float *)retval = excf.retval;
       break;
    }
 case coshl_overflow:
   /* coshl overflow */
   {
      OVERFLOWL; NAMEL = (char *) "coshl";
      ifSVID
      {
        RETVAL_HUGEL;
      }
      else
      {
        RETVAL_HUGE_VALL;
      }
      NOT_MATHERRL {ERRNO_RANGE;}
      *(long double *)retval = excl.retval;
      break;
   }
 case cosh_overflow:
   /* cosh overflow */
   {
      OVERFLOWD; NAMED = (char *) "cosh";
      ifSVID
      {
        RETVAL_HUGED;
      }
      else
      {
        RETVAL_HUGE_VALD;
      }
      NOT_MATHERRD {ERRNO_RANGE;}
      *(double *)retval = exc.retval;
      break;
   }
 case coshf_overflow:
   /* coshf overflow */
   {
      OVERFLOWF; NAMEF = (char *) "coshf";
      ifSVID
      {
        RETVAL_HUGEF;
      }
      else
      {
        RETVAL_HUGE_VALF;
      }
      NOT_MATHERRF {ERRNO_RANGE;}
      *(float *)retval = excf.retval;
      break;
   }
 case sinhl_overflow:
   /* sinhl overflow */
   {
      OVERFLOWL; NAMEL = (char *) "sinhl";
      ifSVID
      {
        if (INPUT_XL > ZEROL_VALUE /*0.0*/) RETVAL_HUGEL;
        else RETVAL_NEG_HUGEL;
      }
      else
      {
        if (INPUT_XL > ZEROL_VALUE /*0.0*/) RETVAL_HUGE_VALL;
        else RETVAL_NEG_HUGE_VALL;
      }
      NOT_MATHERRL {ERRNO_RANGE;}
      *(long double *)retval = excl.retval;
      break;
   }
 case sinh_overflow:
   /* sinh overflow */
   {
      OVERFLOWD; NAMED = (char *) "sinh";
      ifSVID
      {
        if (INPUT_XD > ZEROD_VALUE /*0.0*/) RETVAL_HUGED;
        else RETVAL_NEG_HUGED;
      }
      else
      {
        if (INPUT_XD > ZEROD_VALUE /*0.0*/) RETVAL_HUGE_VALD;
        else RETVAL_NEG_HUGE_VALD;
      }
      NOT_MATHERRD {ERRNO_RANGE;}
      *(double *)retval = exc.retval;
      break;
   }
 case sinhf_overflow:
   /* sinhf overflow */
   {
      OVERFLOWF; NAMEF = (char *) "sinhf";
      ifSVID
      {
        if (INPUT_XF > ZEROF_VALUE /*0.0*/) RETVAL_HUGEF;
        else RETVAL_NEG_HUGEF;
      }
      else
      {
        if (INPUT_XF > ZEROF_VALUE /*0.0*/) RETVAL_HUGE_VALF;
        else RETVAL_NEG_HUGE_VALF;
      }
      NOT_MATHERRF {ERRNO_RANGE;}
      *(float *)retval = excf.retval;
      break;
   }
  case acoshl_lt_one:
    /* acoshl(x < 1) */
    {
       DOMAINL; NAMEL = (char *) "acoshl";
       ifSVID
       {
         NOT_MATHERRL
         {
           WRITEL_ACOSH;
           ERRNO_DOMAIN;
         }
       }
       else
       {
           NOT_MATHERRL {ERRNO_DOMAIN;}
       }
       *(long double *)retval = excl.retval;
       break;
    }
  case acosh_lt_one:
    /* acosh(x < 1) */
    {
       DOMAIND; NAMED = (char *) "acosh";
       ifSVID
       {
         NOT_MATHERRD
         {
          WRITED_ACOSH;
          ERRNO_DOMAIN;
         }
       }
       else
       {
          NOT_MATHERRD {ERRNO_DOMAIN;}
       }
       *(double *)retval = exc.retval;
       break;
    }
  case acoshf_lt_one:
    /* acoshf(x < 1) */
    {
       DOMAINF; NAMEF = (char *) "acoshf";
       ifSVID
       {
         NOT_MATHERRF
         {
           WRITEF_ACOSH;
           ERRNO_DOMAIN;
         }
       }
       else
       {
         NOT_MATHERRF {ERRNO_DOMAIN;}
       }
       *(float *)retval = excf.retval;
       break;
    }
  case atanhl_gt_one:
    /* atanhl(|x| > 1) */
    {
       DOMAINL; NAMEL = (char *) "atanhl";
       ifSVID
       {
         NOT_MATHERRL
         {
           WRITEL_ATANH_GT_ONE;
           ERRNO_DOMAIN;
         }
       }
       else
       {
         NOT_MATHERRL {ERRNO_DOMAIN;}
       }
       break;
    }
  case atanh_gt_one:
    /* atanh(|x| > 1) */
    {
       DOMAIND; NAMED = (char *) "atanh";
       ifSVID
       {
         NOT_MATHERRD
         {
           WRITED_ATANH_GT_ONE;
           ERRNO_DOMAIN;
         }
       }
       else
       {
         NOT_MATHERRD {ERRNO_DOMAIN;}
       }
       break;
    }
  case atanhf_gt_one:
    /* atanhf(|x| > 1) */
    {
       DOMAINF; NAMEF = (char *) "atanhf";
       ifSVID
       {
         NOT_MATHERRF
         {
           WRITEF_ATANH_GT_ONE;
           ERRNO_DOMAIN;
         }
       }
       else
       {
         NOT_MATHERRF {ERRNO_DOMAIN;}
       }
       break;
    }
  case atanhl_eq_one:
    /* atanhl(|x| == 1) */
    {
       SINGL; NAMEL = (char *) "atanhl";
       ifSVID
       {
         NOT_MATHERRL
         {
           WRITEL_ATANH_EQ_ONE;
           ERRNO_DOMAIN;
         }
       }
       else
       {
         NOT_MATHERRL {ERRNO_DOMAIN;}
       }
       break;
    }
  case atanh_eq_one:
    /* atanh(|x| == 1) */
    {
       SINGD; NAMED = (char *) "atanh";
       ifSVID
       {
         NOT_MATHERRD
         {
           WRITED_ATANH_EQ_ONE;
           ERRNO_DOMAIN;
         }
       }
       else
       {
       NOT_MATHERRD {ERRNO_DOMAIN;}
       }
       break;
    }
  case atanhf_eq_one:
    /* atanhf(|x| == 1) */
    {
       SINGF; NAMEF = (char *) "atanhf";
       ifSVID
       {
         NOT_MATHERRF
         {
           WRITEF_ATANH_EQ_ONE;
           ERRNO_DOMAIN;
         }
       }
       else
       {
         NOT_MATHERRF {ERRNO_DOMAIN;}
       }
       break;
    }
  case gammal_overflow:
    /* gammal overflow */
    {
       OVERFLOWL; NAMEL = (char *) "gammal";
       ifSVID
       {
         RETVAL_HUGEL;
       }
       else
       {
         RETVAL_HUGE_VALL;
       }
       NOT_MATHERRL {ERRNO_RANGE;}
       *(long double *)retval = excl.retval;
       break;
    }
  case gamma_overflow:
    /* gamma overflow */
    {
       OVERFLOWD; NAMED = (char *) "gamma";
       ifSVID
       {
         RETVAL_HUGED;
       }
         else
       {
         RETVAL_HUGE_VALD;
       }
       NOT_MATHERRD {ERRNO_RANGE;}
       *(double *)retval = exc.retval;
       break;
    }
  case gammaf_overflow:
    /* gammaf overflow */
    {
       OVERFLOWF; NAMEF = (char *) "gammaf";
       ifSVID
       {
         RETVAL_HUGEF;
       }
       else
       {
         RETVAL_HUGE_VALF;
       }
       NOT_MATHERRF {ERRNO_RANGE;}
       *(float *)retval = excf.retval;
       break;
    }
  case gammal_negative:
    /* gammal -int or 0 */
    {
       SINGL; NAMEL = (char *) "gammal";
       ifSVID
       {
         RETVAL_HUGEL;
         NOT_MATHERRL
         {
            WRITEL_GAMMA_NEGATIVE;
            ERRNO_DOMAIN;
         }
       }
       else
       {
         RETVAL_HUGE_VALL;
         NOT_MATHERRL {ERRNO_DOMAIN;}
       }
       *(long double *)retval = excl.retval;
       break;
    }
  case gamma_negative:
    /* gamma -int or 0 */
    {
       SINGD; NAMED = (char *) "gamma";
       ifSVID
       {
         RETVAL_HUGED;
         NOT_MATHERRD
         {
            WRITED_GAMMA_NEGATIVE;
            ERRNO_DOMAIN;
         }
       }
       else
       {
         RETVAL_HUGE_VALD;
         NOT_MATHERRD {ERRNO_DOMAIN;}
       }
       *(double *)retval = exc.retval;
       break;
    }
  case gammaf_negative:
    /* gammaf -int or 0 */
    {
       SINGF; NAMEF = (char *) "gammaf";
       ifSVID
       {
         RETVAL_HUGEF;
         NOT_MATHERRF
         {
            WRITEF_GAMMA_NEGATIVE;
            ERRNO_DOMAIN;
         }
       }
       else
       {
         RETVAL_HUGE_VALF;
         NOT_MATHERRF {ERRNO_DOMAIN;}
       }
       *(float *)retval = excf.retval;
       break;
    }
  case lgammal_overflow:
    /* lgammal overflow */
    {
       OVERFLOWL; NAMEL = (char *) "lgammal";
       ifSVID
       {
         RETVAL_HUGEL;
       }
       else
       {
         RETVAL_HUGE_VALL;
       }
       NOT_MATHERRL {ERRNO_RANGE;}
       *(long double *)retval = excl.retval;
       break;
    }
  case lgamma_overflow:
    /* lgamma overflow */
    {
       OVERFLOWD; NAMED = (char *) "lgamma";
       ifSVID
       {
         RETVAL_HUGED;
       }
       else
       {
         RETVAL_HUGE_VALD;
       }
       NOT_MATHERRD {ERRNO_RANGE;}
       *(double *)retval = exc.retval;
       break;
    }
  case lgammaf_overflow:
    /* lgammaf overflow */
    {
       OVERFLOWF; NAMEF = (char *) "lgammaf";
       ifSVID
       {
         RETVAL_HUGEF;
       }
       else
       {
         RETVAL_HUGE_VALF;
       }
       NOT_MATHERRF {ERRNO_RANGE;}
       *(float *)retval = excf.retval;
       break;
    }
  case lgammal_negative:
    /* lgammal -int or 0 */
    {
       SINGL; NAMEL = (char *) "lgammal";
       ifSVID
       {
         RETVAL_HUGEL;
         NOT_MATHERRL
         {
           WRITEL_LGAMMA_NEGATIVE;
           ERRNO_DOMAIN;
         }
       }
       else
       {
         RETVAL_HUGE_VALL;
         NOT_MATHERRL {ERRNO_DOMAIN;}
       }
       *(long double *)retval = excl.retval;
       break;
    }
  case lgamma_negative:
    /* lgamma -int or 0 */
    {
       SINGD; NAMED = (char *) "lgamma";
       ifSVID
       {
         RETVAL_HUGED;
         NOT_MATHERRD
         {
           WRITED_LGAMMA_NEGATIVE;
           ERRNO_DOMAIN;
         }
       }
       else
       {
         RETVAL_HUGE_VALD;
         NOT_MATHERRD {ERRNO_DOMAIN;}
       }
       *(double *)retval = exc.retval;
       break;
    }
  case lgammaf_negative:
    /* lgammaf -int or 0 */
    {
       SINGF; NAMEF = (char *) "lgammaf";
       ifSVID
       {
         RETVAL_HUGEF;
         NOT_MATHERRF
         {
           WRITEF_LGAMMA_NEGATIVE;
           ERRNO_DOMAIN;
         }
       }
       else
       {
         RETVAL_HUGE_VALF;
         NOT_MATHERRF {ERRNO_DOMAIN;}
       }
       *(float *)retval = excf.retval;
       break;
    }
  case tgammal_overflow:
    /* tgammal overflow */
    {
       OVERFLOWL; NAMEL = (char *) "tgammal";
       ifSVID
       {
         RETVAL_HUGEL;
       }
       else
       {
         RETVAL_HUGE_VALL;
       }
       NOT_MATHERRL {ERRNO_RANGE;}
       *(long double *)retval = excl.retval;
       break;
    }
  case tgamma_overflow:
    /* tgamma overflow */
    {
       OVERFLOWD; NAMED = (char *) "tgamma";
       ifSVID
       {
         RETVAL_HUGED;
       }
       else
       {
         RETVAL_HUGE_VALD;
       }
       NOT_MATHERRD {ERRNO_RANGE;}
       *(double *)retval = exc.retval;
       break;
    }
  case tgammaf_overflow:
    /* tgammaf overflow */
    {
       OVERFLOWF; NAMEF = (char *) "tgammaf";
       ifSVID
       {
         RETVAL_HUGEF;
       }
       else
       {
         RETVAL_HUGE_VALF;
       }
       NOT_MATHERRF {ERRNO_RANGE;}
       *(float *)retval = excf.retval;
       break;
    }
  case tgammal_negative:
    /* tgammal -int or 0 */
    {
       SINGL; NAMEL = (char *) "tgammal";
       ifSVID
       {
         NOT_MATHERRL
         {
           WRITEL_TGAMMA_NEGATIVE;
           ERRNO_DOMAIN;
         }
       }
       else
       {
         NOT_MATHERRL {ERRNO_DOMAIN;}
       }
       *(long double *)retval = excl.retval;
       break;
    }
  case tgamma_negative:
    /* tgamma -int or 0 */
    {
       SINGD; NAMED = (char *) "tgamma";
       ifSVID
       {
         NOT_MATHERRD
         {
           WRITED_TGAMMA_NEGATIVE;
           ERRNO_DOMAIN;
         }
       }
       else
       {
         NOT_MATHERRD {ERRNO_DOMAIN;}
       }
       *(double *)retval = exc.retval;
       break;
    }
  case tgammaf_negative:
    /* tgammaf -int or 0 */
    {
       SINGF; NAMEF = (char *) "tgammaf";
       ifSVID
       {
         NOT_MATHERRF
         {
           WRITEF_TGAMMA_NEGATIVE;
           ERRNO_DOMAIN;
         }
       }
       else
       {
         NOT_MATHERRF {ERRNO_DOMAIN;}
       }
       *(float *)retval = excf.retval;
       break;
    }
  case j0l_gt_loss:
    /* j0l > loss */
    {
       TLOSSL; NAMEL = (char *) "j0l";
       RETVAL_ZEROL;
       ifSVID
       {
         NOT_MATHERRL
         {
            WRITEL_J0_TLOSS;
            ERRNO_RANGE;
         }
       }
       else
       {
         NOT_MATHERRL {ERRNO_RANGE;}
       }
       *(long double *)retval = excl.retval;
       break;
    }
  case j0_gt_loss:
    /* j0 > loss */
    {
       TLOSSD; NAMED = (char *) "j0";
       RETVAL_ZEROD;
       ifSVID
       {
         NOT_MATHERRD
         {
            WRITED_J0_TLOSS;
            ERRNO_RANGE;
         }
       }
       else
       {
         NOT_MATHERRD {ERRNO_RANGE;}
       }
       *(double*)retval = exc.retval;
       break;
    }
  case j0f_gt_loss:
    /* j0f > loss */
    {
       TLOSSF; NAMEF = (char *) "j0f";
       RETVAL_ZEROF;
       ifSVID
       {
         NOT_MATHERRF
         {
            WRITEF_J0_TLOSS;
            ERRNO_RANGE;
         }
       }
       else
       {
         NOT_MATHERRF {ERRNO_RANGE;}
       }
       *(float*)retval = excf.retval;
       break;
    }
  case j1l_gt_loss:
    /* j1l > loss */
    {
       TLOSSL; NAMEL = (char *) "j1l";
       RETVAL_ZEROL;
       ifSVID
       {
         NOT_MATHERRL
         {
            WRITEL_J1_TLOSS;
            ERRNO_RANGE;
         }
       }
       else
       {
         NOT_MATHERRL {ERRNO_RANGE;}
       }
       *(long double *)retval = excl.retval;
       break;
    }
  case j1_gt_loss:
    /* j1 > loss */
    {
       TLOSSD; NAMED = (char *) "j1";
       RETVAL_ZEROD;
       ifSVID
       {
         NOT_MATHERRD
         {
            WRITED_J1_TLOSS;
            ERRNO_RANGE;
         }
       }
       else
       {
         NOT_MATHERRD {ERRNO_RANGE;}
       }
       *(double*)retval = exc.retval;
       break;
    }
  case j1f_gt_loss:
    /* j1f > loss */
    {
       TLOSSF; NAMEF = (char *) "j1f";
       RETVAL_ZEROF;
       ifSVID
       {
         NOT_MATHERRF
         {
            WRITEF_J1_TLOSS;
            ERRNO_RANGE;
         }
       }
       else
       {
         NOT_MATHERRF {ERRNO_RANGE;}
       }
       *(float*)retval = excf.retval;
       break;
    }
  case jnl_gt_loss:
    /* jnl > loss */
    {
       TLOSSL; NAMEL = (char *) "jnl";
       RETVAL_ZEROL;
       ifSVID
       {
         NOT_MATHERRL
         {
            WRITEL_JN_TLOSS;
            ERRNO_RANGE;
         }
       }
       else
       {
         NOT_MATHERRL {ERRNO_RANGE;}
       }
       *(long double *)retval = excl.retval;
       break;
    }
  case jn_gt_loss:
    /* jn > loss */
    {
       TLOSSD; NAMED = (char *) "jn";
       RETVAL_ZEROD;
       ifSVID
       {
         NOT_MATHERRD
         {
            WRITED_JN_TLOSS;
            ERRNO_RANGE;
         }
       }
       else
       {
         NOT_MATHERRD {ERRNO_RANGE;}
       }
       *(double*)retval = exc.retval;
       break;
    }
  case jnf_gt_loss:
    /* jnf > loss */
    {
       TLOSSF; NAMEF = (char *) "jnf";
       RETVAL_ZEROF;
       ifSVID
       {
         NOT_MATHERRF
         {
            WRITEF_JN_TLOSS;
            ERRNO_RANGE;
         }
       }
       else
       {
         NOT_MATHERRF {ERRNO_RANGE;}
       }
       *(float*)retval = excf.retval;
       break;
    }
  case y0l_gt_loss:
    /* y0l > loss */
    {
       TLOSSL; NAMEL = (char *) "y0l";
       RETVAL_ZEROL;
       ifSVID
       {
         NOT_MATHERRL
         {
            WRITEL_Y0_TLOSS;
            ERRNO_RANGE;
         }
       }
       else
       {
         NOT_MATHERRL {ERRNO_RANGE;}
       }
       *(long double *)retval = excl.retval;
       break;
    }
  case y0_gt_loss:
    /* y0 > loss */
    {
       TLOSSD; NAMED = (char *) "y0";
       RETVAL_ZEROD;
       ifSVID
       {
         NOT_MATHERRD
         {
            WRITED_Y0_TLOSS;
            ERRNO_RANGE;
         }
       }
       else
       {
         NOT_MATHERRD {ERRNO_RANGE;}
       }
       *(double*)retval = exc.retval;
       break;
    }
  case y0f_gt_loss:
    /* y0f > loss */
    {
       TLOSSF; NAMEF = (char *) "y0f";
       RETVAL_ZEROF;
       ifSVID
       {
         NOT_MATHERRF
         {
            WRITEF_Y0_TLOSS;
            ERRNO_RANGE;
         }
       }
       else
       {
         NOT_MATHERRF {ERRNO_RANGE;}
       }
       *(float*)retval = excf.retval;
       break;
    }
  case y0l_zero:
    /* y0l(0) */
    {
       DOMAINL; NAMEL = (char *) "y0l";
       ifSVID
       {
         RETVAL_NEG_HUGEL;
         NOT_MATHERRL
         {
           WRITEL_Y0_ZERO;
           ERRNO_DOMAIN;
         }
       }
       else
       {
         RETVAL_NEG_HUGE_VALL;
         NOT_MATHERRL {ERRNO_DOMAIN;}
       }
       *(long double *)retval = excl.retval;
       break;
    }
  case y0_zero:
    /* y0(0) */
    {
       DOMAIND; NAMED = (char *) "y0";
       ifSVID
       {
         RETVAL_NEG_HUGED;
         NOT_MATHERRD
         {
           WRITED_Y0_ZERO;
           ERRNO_DOMAIN;
         }
       }
       else
       {
         RETVAL_NEG_HUGE_VALD;
         NOT_MATHERRD {ERRNO_DOMAIN;}
       }
       *(double *)retval = exc.retval;
       break;
    }
  case y0f_zero:
    /* y0f(0) */
    {
       DOMAINF; NAMEF = (char *) "y0f";
       ifSVID
       {
         RETVAL_NEG_HUGEF;
         NOT_MATHERRF
         {
           WRITEF_Y0_ZERO;
           ERRNO_DOMAIN;
         }
       }
       else
       {
         RETVAL_NEG_HUGE_VALF;
         NOT_MATHERRF {ERRNO_DOMAIN;}
       }
       *(float *)retval = excf.retval;
       break;
    }
  case y1l_gt_loss:
    /* y1l > loss */
    {
       TLOSSL; NAMEL = (char *) "y1l";
       RETVAL_ZEROL;
       ifSVID
       {
         NOT_MATHERRL
         {
            WRITEL_Y1_TLOSS;
            ERRNO_RANGE;
         }
       }
       else
       {
         NOT_MATHERRL {ERRNO_RANGE;}
       }
       *(long double *)retval = excl.retval;
       break;
    }
  case y1_gt_loss:
    /* y1 > loss */
    {
       TLOSSD; NAMED = (char *) "y1";
       RETVAL_ZEROD;
       ifSVID
       {
         NOT_MATHERRD
         {
            WRITED_Y1_TLOSS;
            ERRNO_RANGE;
         }
       }
       else
       {
         NOT_MATHERRD {ERRNO_RANGE;}
       }
       *(double*)retval = exc.retval;
       break;
    }
  case y1f_gt_loss:
    /* y1f > loss */
    {
       TLOSSF; NAMEF = (char *) "y1f";
       RETVAL_ZEROF;
       ifSVID
       {
         NOT_MATHERRF
         {
            WRITEF_Y1_TLOSS;
            ERRNO_RANGE;
         }
       }
       else
       {
         NOT_MATHERRF {ERRNO_RANGE;}
       }
       *(float*)retval = excf.retval;
       break;
    }
  case y1l_zero:
    /* y1l(0) */
    {
       DOMAINL; NAMEL = (char *) "y1l";
       ifSVID
       {
         RETVAL_NEG_HUGEL;
         NOT_MATHERRL
         {
           WRITEL_Y1_ZERO;
           ERRNO_DOMAIN;
         }
       }
       else
       {
         RETVAL_NEG_HUGE_VALL;
         NOT_MATHERRL {ERRNO_DOMAIN;}
       }
       *(long double *)retval = excl.retval;
       break;
    }
  case y1_zero:
    /* y1(0) */
    {
       DOMAIND; NAMED = (char *) "y1";
       ifSVID
       {
         RETVAL_NEG_HUGED;
         NOT_MATHERRD
         {
           WRITED_Y1_ZERO;
           ERRNO_DOMAIN;
         }
       }
       else
       {
         RETVAL_NEG_HUGE_VALD;
         NOT_MATHERRD {ERRNO_DOMAIN;}
       }
       *(double *)retval = exc.retval;
       break;
    }
  case y1f_zero:
    /* y1f(0) */
    {
       DOMAINF; NAMEF = (char *) "y1f";
       ifSVID
       {
         RETVAL_NEG_HUGEF;
         NOT_MATHERRF
         {
           WRITEF_Y1_ZERO;
           ERRNO_DOMAIN;
         }
       }
       else
       {
         RETVAL_NEG_HUGE_VALF;
         NOT_MATHERRF {ERRNO_DOMAIN;}
       }
       *(float *)retval = excf.retval;
       break;
    }
  case ynl_gt_loss:
    /* ynl > loss */
    {
       TLOSSL; NAMEL = (char *) "ynl";
       RETVAL_ZEROL;
       ifSVID
       {
         NOT_MATHERRL
         {
            WRITEL_YN_TLOSS;
            ERRNO_RANGE;
         }
       }
       else
       {
         NOT_MATHERRL {ERRNO_RANGE;}
       }
       *(long double *)retval = excl.retval;
       break;
    }
  case yn_gt_loss:
    /* yn > loss */
    {
       TLOSSD; NAMED = (char *) "yn";
       RETVAL_ZEROD;
       ifSVID
       {
         NOT_MATHERRD
         {
            WRITED_YN_TLOSS;
            ERRNO_RANGE;
         }
       }
       else
       {
         NOT_MATHERRD {ERRNO_RANGE;}
       }
       *(double*)retval = exc.retval;
       break;
    }
  case ynf_gt_loss:
    /* ynf > loss */
    {
       TLOSSF; NAMEF = (char *) "ynf";
       RETVAL_ZEROF;
       ifSVID
       {
         NOT_MATHERRF
         {
            WRITEF_YN_TLOSS;
            ERRNO_RANGE;
         }
       }
       else
       {
         NOT_MATHERRF {ERRNO_RANGE;}
       }
       *(float*)retval = excf.retval;
       break;
    }
  case ynl_zero:
    /* ynl(0) */
    {
       DOMAINL; NAMEL = (char *) "ynl";
       ifSVID
       {
         RETVAL_NEG_HUGEL;
         NOT_MATHERRL
         {
           WRITEL_YN_ZERO;
           ERRNO_DOMAIN;
         }
       }
       else
       {
         RETVAL_NEG_HUGE_VALL;
         NOT_MATHERRL {ERRNO_DOMAIN;}
       }
       *(long double *)retval = excl.retval;
       break;
    }
  case yn_zero:
    /* yn(0) */
    {
       DOMAIND; NAMED = (char *) "yn";
       ifSVID
       {
         RETVAL_NEG_HUGED;
         NOT_MATHERRD
         {
           WRITED_YN_ZERO;
           ERRNO_DOMAIN;
         }
       }
       else
       {
         RETVAL_NEG_HUGE_VALD;
         NOT_MATHERRD {ERRNO_DOMAIN;}
       }
       *(double *)retval = exc.retval;
       break;
    }
  case ynf_zero:
    /* ynf(0) */
    {
       DOMAINF; NAMEF = (char *) "ynf";
       ifSVID
       {
         RETVAL_NEG_HUGEF;
         NOT_MATHERRF
         {
           WRITEF_YN_ZERO;
           ERRNO_DOMAIN;
         }
       }
       else
       {
         RETVAL_NEG_HUGE_VALF;
         NOT_MATHERRF {ERRNO_DOMAIN;}
       }
       *(float *)retval = excf.retval;
       break;
    }
  case y0l_negative:
    /* y0l(x<0) */
    {
       DOMAINL; NAMEL = (char *) "y0l";
       ifSVID
       {
         RETVAL_NEG_HUGEL;
         NOT_MATHERRL
         {
           WRITEL_Y0_NEGATIVE;
           ERRNO_DOMAIN;
         }
       }
       else
       {
         RETVAL_NEG_HUGE_VALL;
         NOT_MATHERRL {ERRNO_DOMAIN;}
       }
       *(long double *)retval = excl.retval;
       break;
    }
  case y0_negative:
    /* y0(x<0) */
    {
       DOMAIND; NAMED = (char *) "y0";
       ifSVID
       {
         RETVAL_NEG_HUGED;
         NOT_MATHERRD
         {
           WRITED_Y0_NEGATIVE;
           ERRNO_DOMAIN;
         }
       }
       else
       {
         RETVAL_NEG_HUGE_VALD;
         NOT_MATHERRD {ERRNO_DOMAIN;}
       }
       *(double *)retval = exc.retval;
       break;
    }
  case y0f_negative:
    /* y0f(x<0) */
    {
       DOMAINF; NAMEF = (char *) "y0f";
       ifSVID
       {
         RETVAL_NEG_HUGEF;
         NOT_MATHERRF
         {
           WRITEF_Y0_NEGATIVE;
           ERRNO_DOMAIN;
         }
       }
       else
       {
         RETVAL_NEG_HUGE_VALF;
         NOT_MATHERRF {ERRNO_DOMAIN;}
       }
       *(float *)retval = excf.retval;
       break;
    }
  case y1l_negative:
    /* y1l(x<0) */
    {
       DOMAINL; NAMEL = (char *) "y1l";
       ifSVID
       {
         RETVAL_NEG_HUGEL;
         NOT_MATHERRL
         {
           WRITEL_Y1_NEGATIVE;
           ERRNO_DOMAIN;
         }
       }
       else
       {
         RETVAL_NEG_HUGE_VALL;
         NOT_MATHERRL {ERRNO_DOMAIN;}
       }
       *(long double *)retval = excl.retval;
       break;
    }
  case y1_negative:
    /* y1(x<0) */
    {
       DOMAIND; NAMED = (char *) "y1";
       ifSVID
       {
         RETVAL_NEG_HUGED;
         NOT_MATHERRD
         {
           WRITED_Y1_NEGATIVE;
           ERRNO_DOMAIN;
         }
       }
       else
       {
         RETVAL_NEG_HUGE_VALD;
         NOT_MATHERRD {ERRNO_DOMAIN;}
       }
       *(double *)retval = exc.retval;
       break;
    }
  case y1f_negative:
    /* y1f(x<0) */
    {
       DOMAINF; NAMEF = (char *) "y1f";
       ifSVID
       {
         RETVAL_NEG_HUGEF;
         NOT_MATHERRF
         {
           WRITEF_Y1_NEGATIVE;
           ERRNO_DOMAIN;
         }
       }
       else
       {
         RETVAL_NEG_HUGE_VALF;
         NOT_MATHERRF {ERRNO_DOMAIN;}
       }
       *(float *)retval = excf.retval;
       break;
    }
  case ynl_negative:
    /* ynl(x<0) */
    {
       DOMAINL; NAMEL = (char *) "ynl";
       ifSVID
       {
         RETVAL_NEG_HUGEL;
         NOT_MATHERRL
         {
          WRITEL_YN_NEGATIVE;
          ERRNO_DOMAIN;
         }
       }
       else
       {
         RETVAL_NEG_HUGE_VALL;
         NOT_MATHERRL {ERRNO_DOMAIN;}
       }
       *(long double *)retval = excl.retval;
       break;
    }
  case yn_negative:
    /* yn(x<0) */
    {
       DOMAIND; NAMED = (char *) "yn";
       ifSVID
       {
         RETVAL_NEG_HUGED;
         NOT_MATHERRD
         {
           WRITED_YN_NEGATIVE;
           ERRNO_DOMAIN;
         }
       }
       else
       {
         RETVAL_NEG_HUGE_VALD;
         NOT_MATHERRD {ERRNO_DOMAIN;}
       }
       *(double *)retval = exc.retval;
       break;
    }
  case ynf_negative:
    /* ynf(x<0) */
    {
       DOMAINF; NAMEF = (char *) "ynf";
       ifSVID
       {
         RETVAL_NEG_HUGEF;
         NOT_MATHERRF
         {
           WRITEF_YN_NEGATIVE;
           ERRNO_DOMAIN;
         }
       }
       else
       {
         RETVAL_NEG_HUGE_VALF;
         NOT_MATHERRF {ERRNO_DOMAIN;}
       }
       *(float *)retval = excf.retval;
       break;
    }
  case fmodl_by_zero:
    /* fmodl(x,0) */
    {
       DOMAINL; NAMEL = (char *) "fmodl";
       ifSVID
       {
            *(long double *)retval = *(long double *)arg1;
            NOT_MATHERRL
            {
              WRITEL_FMOD;
              ERRNO_DOMAIN;
            }
       }
       else
       { /* NaN already computed */
            NOT_MATHERRL {ERRNO_DOMAIN;}
       }
       *(long double *)retval = excl.retval;
       break;
    }
  case fmod_by_zero:
    /* fmod(x,0) */
    {
       DOMAIND; NAMED = (char *) "fmod";
       ifSVID
       {
         *(double *)retval = *(double *)arg1;
         NOT_MATHERRD
         {
           WRITED_FMOD;
           ERRNO_DOMAIN;
         }
       }
       else
       { /* NaN already computed */
         NOT_MATHERRD {ERRNO_DOMAIN;}
       }
       *(double *)retval = exc.retval;
       break;
    }
  case fmodf_by_zero:
    /* fmodf(x,0) */
    {
       DOMAINF; NAMEF = (char *) "fmodf";
       ifSVID
       {
         *(float *)retval = *(float *)arg1;
         NOT_MATHERRF
         {
           WRITEF_FMOD;
           ERRNO_DOMAIN;
         }
       }
       else
       {
         NOT_MATHERRF {ERRNO_DOMAIN;}
       }
       *(float *)retval = excf.retval;
       break;
    }
  case remainderl_by_zero:
    /* remainderl(x,0) */
    {
       DOMAINL; NAMEL = (char *) "remainderl";
       ifSVID
       {
          NOT_MATHERRL
          {
            WRITEL_REM;
            ERRNO_DOMAIN;
          }
       }
       else
       { /* NaN already computed */
            NOT_MATHERRL {ERRNO_DOMAIN;}
       }
       *(long double *)retval = excl.retval;
       break;
    }
  case remainder_by_zero:
    /* remainder(x,0) */
    {
       DOMAIND; NAMED = (char *) "remainder";
       ifSVID
       {
         NOT_MATHERRD
         {
           WRITED_REM;
           ERRNO_DOMAIN;
         }
       }
       else
       { /* NaN already computed */
         NOT_MATHERRD {ERRNO_DOMAIN;}
       }
       *(double *)retval = exc.retval;
       break;
    }
  case remainderf_by_zero:
    /* remainderf(x,0) */
    {
       DOMAINF; NAMEF = (char *) "remainderf";
       ifSVID
       {
         NOT_MATHERRF
         {
           WRITEF_REM;
           ERRNO_DOMAIN;
         }
       }
       else
       {
         NOT_MATHERRF {ERRNO_DOMAIN;}
       }
       *(float *)retval = excf.retval;
       break;
    }
  default:
    /* We don't want to abort () since SVID doesn't cover all math
       library functions.  */
    break;
   }
   return;
   }
}
