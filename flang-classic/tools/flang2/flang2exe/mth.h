/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

#ifndef MTH_H_
#define MTH_H_

/*  mth.h - 'parameterize' the names of the __mth_i/__fmth_i ... functions.
 *          Macros are not used for the new naming conventions, i.e.,
 *             _f[sv][sd]_BASE, _g[sv][sdcz]_BASE[L]
 *
 *  New ...
 *  Create the function representing a math routine:
 *  Name is 
 *     __<ftype><data type>_<name>_<vectlen><mask>
 *
 * <ftype>     : f - fastmath (default)
 *               r - relaxed math (-Mfprelaxed ...)
 *               p - precise math (-Kieee)
 * <data type> : s - single precision
 *               d - double precision
 *               c - single precision complex
 *               z - double precision complex
 * <name>      : exp, log, log10, pow, powi, powk, powi1, powk1, sin, cos, 
 *               tan, asin, acos, atan, sinh, cosh, tanh, atan2, 
 * <vectlen>   : 1 (scalar), 2, 4, 8, 16
 * <mask>      : m or null
 *
exp
log
log10
pow
powi
powk
powi1
powk1
sin
cos
tan
asin
acos
atan
sinh
cosh
tanh
atan2
div
mod
floor
ceil
aint
 */

typedef enum MTH_FN {
  MTH_acos,
  MTH_asin,
  MTH_atan,
  MTH_atan2,
  MTH_cos,
  MTH_cosh,
  MTH_div,
  MTH_exp,
  MTH_log,
  MTH_log10,
  MTH_pow,
  MTH_powi,
  MTH_powk,
  MTH_powi1,
  MTH_powk1,
  MTH_sin,
  MTH_sincos,
  MTH_sinh,
  MTH_sqrt,
  MTH_tan,
  MTH_tanh,
  MTH_mod,
  MTH_floor,
  MTH_ceil,
  MTH_aint
} MTH_FN;

#define MTH_I_DFIXK "__mth_i_dfixk"

#define MTH_I_DFIXUK "__mth_i_dfixuk"

#define MTH_I_DFLOATK "__mth_i_dfloatk"
#define MTH_I_DFLOATUK "__mth_i_dfloatuk"
#define MTH_I_DRSQRT "__mth_i_drsqrt"
#define MTH_I_FIXK "__mth_i_fixk"
#define MTH_I_FIXUK "__mth_i_fixuk"
#define MTH_I_FLOATK "__mth_i_floatk"
#define MTH_I_FLOATUK "__mth_i_floatuk"
#define MTH_I_FRSQRT "__mth_i_frsqrt"
#define MTH_I_KCMP "__mth_i_kcmp"
#define MTH_I_KICSHFT "__mth_i_kicshft"
#define MTH_I_UKICSHFT "__mth_i_ukicshft"
#define MTH_I_KULSHIFT "__mth_i_klshift"
#define MTH_I_KRSHIFT "__mth_i_krshift"
#define MTH_I_KURSHIFT "__mth_i_kurshift"

#define MTH_I_NINT "__mth_i_nint"
#define MTH_I_KNINT "__mth_i_knint"
#define MTH_I_IDNINT "__mth_i_idnint"
#define MTH_I_KIDNINT "__mth_i_kidnnt"
#define MTH_I_IQNINT "__mth_i_iqnint"
#define MTH_I_KIQNNT "__mth_i_kiqnint"

#define MTH_I_FLOATU "__mth_i_floatu"
#define MTH_I_DFLOATU "__mth_i_dfloatu"

#define MTH_I_ATAN "__mth_i_atan"
#define MTH_I_DATAN "__mth_i_datan"
#define MTH_I_QATAN "__mth_i_qatan"
#define MTH_I_SIN "__mth_i_sin"
#define MTH_I_COS "__mth_i_cos"
#define MTH_I_SINCOS "__mth_i_sincos"
#define MTH_I_TAN "__mth_i_tan"
#define MTH_I_QTAN "__mth_i_qtan"
#define MTH_I_DSIN "__mth_i_dsin"
#define MTH_I_DCOS "__mth_i_dcos"
#define MTH_I_DSINCOS "__mth_i_dsincos"
#define MTH_I_DTAN "__mth_i_dtan"
#define MTH_I_RPOWI "__mth_i_rpowi"
#define MTH_I_RPOWK "__mth_i_rpowk"
#define MTH_I_RPOWF "__mth_i_rpowr"
#define MTH_I_DPOWI "__mth_i_dpowi"
#define MTH_I_DPOWK "__mth_i_dpowk"
#define MTH_I_DPOWD "__mth_i_dpowd"
#define MTH_I_QPOWI "__mth_i_qpowi"
#define MTH_I_QPOWK "__mth_i_qpowk"
#define MTH_I_QPOWQ "__mth_i_qpowq"
#define MTH_I_FSIGN "__mth_i_sign"
#define MTH_I_DSIGN "__mth_i_dsign"
#define MTH_I_EXP "__mth_i_exp"
#define MTH_I_DEXP "__mth_i_dexp"
#define MTH_I_QEXP "__mth_i_qexp"
#define MTH_I_ALOG "__mth_i_alog"
#define MTH_I_DLOG "__mth_i_dlog"
#define MTH_I_ALOG10 "__mth_i_alog10"
#define MTH_I_DLOG10 "__mth_i_dlog10"
#define MTH_I_AMOD "__mth_i_amod"
#define MTH_I_DMOD "__mth_i_dmod"
#define MTH_I_QMOD "__mth_i_qmod"
#define MTH_I_SINH "__mth_i_sinh"
#define MTH_I_COSH "__mth_i_cosh"
#define MTH_I_TANH "__mth_i_tanh"
#define MTH_I_DSINH "__mth_i_dsinh"
#define MTH_I_DCOSH "__mth_i_dcosh"
#define MTH_I_DTANH "__mth_i_dtanh"
#define MTH_I_FFLOOR "__mth_i_ffloor"
#define MTH_I_DFLOOR "__mth_i_dfloor"
#define MTH_I_QFLOOR "__mth_i_qfloor"
#define MTH_I_FCEIL "__mth_i_fceil"
#define MTH_I_DCEIL "__mth_i_dceil"
#define MTH_I_QCEIL "__mth_i_qceil"
#define MTH_I_AINT "__mth_i_aint"
#define MTH_I_DINT "__mth_i_dint"
#define MTH_I_QINT "__mth_i_qint"
#define MTH_I_QANINT "__mth_i_qanint"
#define MTH_I_DQANINT "__mth_i_dqanint"
#define MTH_I_AQANINT "__mth_i_aqanint"

#define MTH_I_JN "__mth_i_bessel_jn"
#define MTH_I_DJN "__mth_i_dbessel_jn"
#define MTH_I_YN "__mth_i_bessel_yn"
#define MTH_I_DYN "__mth_i_dbessel_yn"

#define FMTH_I_RPOWF "__fmth_i_rpowr"
#define FMTH_I_DPOWD "__fmth_i_dpowd"
#define FMTH_I_EXP "__fmth_i_exp"
#define FMTH_I_DEXP "__fmth_i_dexp"
#define FMTH_I_ALOG "__fmth_i_alog"
#define FMTH_I_DLOG "__fmth_i_dlog"
#define FMTH_I_ALOG10 "__fmth_i_alog10"
#define FMTH_I_DLOG10 "__fmth_i_dlog10"
#define FMTH_I_CBRT "__fmth_i_cbrt"
#define FMTH_I_AMOD "__fmth_i_amod"
#define FMTH_I_DMOD "__fmth_i_dmod"
#define FMTH_I_SIN "__fmth_i_sin"
#define FMTH_I_DSIN "__fmth_i_dsin"
#define MTH_I_QSIN "__mth_i_qsin"
#define FMTH_I_COS "__fmth_i_cos"
#define FMTH_I_DCOS "__fmth_i_dcos"
#define MTH_I_QCOS "__mth_i_qcos"
#define FMTH_I_SINCOS "__fmth_i_sincos"
#define FMTH_I_DSINCOS "__fmth_i_dsincos"
#define FMTH_I_SINH "__fmth_i_sinh"
#define FMTH_I_COSH "__fmth_i_cosh"
#define FMTH_I_DSINH "__fmth_i_dsinh"
#define FMTH_I_DCOSH "__fmth_i_dcosh"
#define FMTH_I_TAN "__fmth_i_tan"
#define FMTH_I_DTAN "__fmth_i_dtan"
#define FMTH_I_CSDIV "__fsc_div"
#define FMTH_I_CDDIV "__fsz_div"

#define MTH_I_ACOS "__mth_i_acos"
#define MTH_I_ASIN "__mth_i_asin"
#define MTH_I_ATAN2 "__mth_i_atan2"
#define MTH_I_DADD "__mth_i_dadd"
#define MTH_I_DACOS "__mth_i_dacos"
#define MTH_I_QACOS "__mth_i_qacos"
#define MTH_I_DASIN "__mth_i_dasin"
#define MTH_I_QASIN "__mth_i_qasin"
#define MTH_I_DATAN2 "__mth_i_datan2"
#define MTH_I_QATAN2 "__mth_i_qatan2"
#define MTH_I_DBLE "__mth_i_dble"
#define MTH_I_DCMP "__mth_i_dcmp"
#define MTH_I_DDIV "__mth_i_ddiv"
#define MTH_I_IPOWI "__mth_i_ipowi"
#define MTH_I_KPOWI "__mth_i_kpowi"
#define MTH_I_KPOWK "__mth_i_kpowk"
#define MTH_I_DFIX "__mth_i_dfix"
#define MTH_I_DFLOAT "__mth_i_dfloat"
#define MTH_I_DFLOATK "__mth_i_dfloatk"
#define MTH_I_DFLOATUK "__mth_i_dfloatuk"
#define MTH_I_DMUL "__mth_i_dmul"
#define MTH_I_DSQRT "__mth_i_dsqrt"
#define MTH_I_QSQRT "__mth_i_qsqrt"
#define MTH_I_DSUB "__mth_i_dsub"
#define MTH_I_FADD "__mth_i_fadd"
#define MTH_I_FCMP "__mth_i_fcmp"
#define MTH_I_FDIV "__mth_i_fdiv"
#define MTH_I_FIX "__mth_i_fix"
#define MTH_I_FLOAT "__mth_i_float"
#define MTH_I_FLOATK "__mth_i_floatk"
#define MTH_I_FLOATUK "__mth_i_floatuk"
#define MTH_I_FMUL "__mth_i_fmul"
#define MTH_I_FSUB "__mth_i_fsub"
#define MTH_I_IDIV "__mth_i_idiv"
#define MTH_I_IMOD "__mth_i_imod"
#define MTH_I_KADD "__mth_i_kadd"
#define MTH_I_KCMP "__mth_i_kcmp"
#define MTH_I_KCMPZ "__mth_i_kcmpz"
#define MTH_I_KDIV "__mth_i_kdiv"
#define MTH_I_KMUL "__mth_i_kmul"
#define MTH_I_KMAX "__mth_i_kmax"
#define MTH_I_KMIN "__mth_i_kmin"
#define MTH_I_KSUB "__mth_i_ksub"
#define MTH_I_KUCMP "__mth_i_kucmp"
#define MTH_I_KUCMPZ "__mth_i_kucmpz"
#define MTH_I_LOG "__mth_i_log"
#define MTH_I_QLOG "__mth_i_qlog"
#define MTH_I_LOG10 "__mth_i_log10"
#define MTH_I_QIDIV "__mth_i_qidiv"
#define MTH_I_RDIV "__mth_i_rdiv"
#define MTH_I_SNGL "__mth_i_sngl"
#define MTH_I_SQRT "__mth_i_sqrt"
#define MTH_I_UICMP "__mth_i_uicmp"
#define MTH_I_UIDIV "__mth_i_uidiv"
#define MTH_I_UIMOD "__mth_i_uimod"
#define MTH_I_UKDIV "__mth_i_ukdiv"
#define MTH_I_KULSHIFT "__mth_i_klshift"
#define MTH_I_KRSHIFT "__mth_i_krshift"
#define MTH_I_KURSHIFT "__mth_i_kurshift"
#define MTH_I_ILEADZI "__mth_i_ileadzi"
#define MTH_I_ILEADZ "__mth_i_ileadz"
#define MTH_I_KLEADZ "__mth_i_kleadz"
#define MTH_I_ITRAILZI "__mth_i_itrailzi"
#define MTH_I_ITRAILZ "__mth_i_itrailz"
#define MTH_I_KTRAILZ "__mth_i_ktrailz"
#define MTH_I_IPOPCNTI "__mth_i_ipopcnti"
#define MTH_I_IPOPCNT "__mth_i_ipopcnt"
#define MTH_I_KPOPCNT "__mth_i_kpopcnt"
#define MTH_I_IPOPPARI "__mth_i_ipoppari"
#define MTH_I_IPOPPAR "__mth_i_ipoppar"
#define MTH_I_KPOPPAR "__mth_i_kpoppar"

#define MTH_I_KBITS "ftn_i_kibits"
#define MTH_I_KBSET "ftn_i_kibset"
#define MTH_I_KBTEST "ftn_i_bktest"
#define MTH_I_KBCLR "ftn_i_kibclr"

#endif // MTH_H_
