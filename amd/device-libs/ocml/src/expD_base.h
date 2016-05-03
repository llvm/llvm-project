
#include "mathD.h"

//   Algorithm:
//
//   e^x = 2^(x/ln(2)) = 2^(x*(64/ln(2))/64)
//
//   x*(64/ln(2)) = n + f, |f| <= 0.5, n is integer
//   n = 64*m + j,   0 <= j < 64
//
//   e^x = 2^((64*m + j + f)/64)
//       = (2^m) * (2^(j/64)) * 2^(f/64)
//       = (2^m) * (2^(j/64)) * e^(f*(ln(2)/64))
//
//   f = x*(64/ln(2)) - n
//   r = f*(ln(2)/64) = x - n*(ln(2)/64)
//
//   e^x = (2^m) * (2^(j/64)) * e^r
//
//   (2^(j/64)) is precomputed
//
//   e^r = 1 + r + (r^2)/2! + (r^3)/3! + (r^4)/4! + (r^5)/5!
//   e^r = 1 + q
//
//   q = r + (r^2)/2! + (r^3)/3! + (r^4)/4! + (r^5)/5!
//
//   e^x = (2^m) * ( (2^(j/64)) + q*(2^(j/64)) ) 

PUREATTR double
#if defined COMPILING_EXP2
MATH_MANGLE(exp2)(double x)
#elif defined COMPILING_EXP10
MATH_MANGLE(exp10)(double x)
#else
MATH_MANGLE(exp)(double x)
#endif
{
    USE_TABLE(double2, p_tbl, M64_EXP_EP);

#if defined(COMPILING_EXP2)
    const double X_MAX = 1024.0;
    const double X_MIN = -1074;
#elif defined(COMPILING_EXP10)
    const double X_MAX = 0x1.34413509f79ffp+8; // 1024*ln(2)/ln(10)
    const double X_MIN = -0x1.434e6420f4374p+8; // -1074*ln(2)/ln(10)
#else
    const double X_MAX = 0x1.62e42fefa39efp+9; // 1024*ln(2)
    const double X_MIN = -0x1.74910d52d3051p+9; // -1075*ln(2)
#endif

#if defined(COMPILING_EXP2)
    const double R_64 = 64.0;
    const double R_1_BY_64 = 1.0 / 64.0;
    const double R_LN2 = 0x1.62e42fefa39efp-1; // ln(2)
#elif defined(COMPILING_EXP10)
    const double R_64_BY_LOG10_2 = 0x1.a934f0979a371p+7; // 64*ln(10)/ln(2)
    const double R_LOG10_2_BY_64_LD = 0x1.3441350000000p-8; // head ln(2)/(64*ln(10))  
    const double R_LOG10_2_BY_64_TL = 0x1.3ef3fde623e25p-37; // tail ln(2)/(64*ln(10))
    const double R_LN10 = 0x1.26bb1bbb55516p+1; // ln(10)
#else
    const double R_64_BY_LOG2 = 0x1.71547652b82fep+6; // 64/ln(2)
    const double R_LOG2_BY_64_LD = 0x1.62e42fefa0000p-7; // head ln(2)/64
    const double R_LOG2_BY_64_TL = 0x1.cf79abc9e3b39p-46; // tail ln(2)/64
#endif

#if defined(COMPILING_EXP2)
    int n = (int)(x * R_64);
#elif defined(COMPILING_EXP10)
    int n = (int)(x * R_64_BY_LOG10_2);
#else
    int n = (int)(x * R_64_BY_LOG2);
#endif

    double dn = (double)n;

    int j = n & 0x3f;
    int m = n >> 6;

#if defined(COMPILING_EXP2)
    double r = R_LN2 * MATH_MAD(-R_1_BY_64, dn, x);
#elif defined(COMPILING_EXP10)
    double r = R_LN10 * MATH_MAD(-R_LOG10_2_BY_64_TL, dn, MATH_MAD(-R_LOG10_2_BY_64_LD, dn, x));
#else
    double r = MATH_MAD(-R_LOG2_BY_64_TL, dn, MATH_MAD(-R_LOG2_BY_64_LD, dn, x));
#endif

    // 6 term tail of Taylor expansion of e^r
    double z2 = r * MATH_MAD(r,
	                MATH_MAD(r,
		            MATH_MAD(r,
			        MATH_MAD(r,
			            MATH_MAD(r, 0x1.6c16c16c16c17p-10, 0x1.1111111111111p-7),
			            0x1.5555555555555p-5),
			        0x1.5555555555555p-3),
		            0x1.0000000000000p-1),
		        1.0);

    double2 tv = p_tbl[j];
    z2 = MATH_MAD(tv.s0 + tv.s1, z2, tv.s1) + tv.s0;

    if (AMD_OPT()) {
        z2 = BUILTIN_FLDEXP_F64(z2, m);
    } else {
        double ss = as_double(0x1L << (m + 1074));
        double sn = as_double((long)(m + EXPBIAS_DP64) << EXPSHIFTBITS_DP64);
        z2 *= m < -1022 ? ss : sn;
    }

    if (!FINITE_ONLY_OPT()) {
        z2 = BUILTIN_CLASS_F64(x, CLASS_SNAN|CLASS_QNAN) ? x : z2;
        z2 = x > X_MAX ? as_double(PINFBITPATT_DP64) : z2;
    }

    z2 = x < X_MIN ? 0.0 : z2;

    return z2;
}

