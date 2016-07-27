/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathD.h"
#undef AVOID_CONTROL_FLOW

CONSTATTR double
MATH_MANGLE(atan)(double x)
{
    const double piby2 = 0x1.921fb54442d18p+0;

    double v = BUILTIN_ABS_F64(x);

#if !defined AVOID_CONTROL_FLOW
    double a, b, clo, chi;
    if (v <= 0x1.cp-2) { // v < 7/16
        a = v;
        b = 1.0;
        chi = 0.0;
        clo = 0.0;
    } else if (v <= 0x1.6p-1) { // 11/16 > v > 7/16
        a = MATH_MAD(v, 2.0, -1.0);
        b = 2.0 + v;
        // (chi + clo) = arctan(0.5)
        chi = 0x1.dac670561bb4fp-2;
        clo = 0x1.a2b7f222f65e0p-56;
    } else if (v <= 0x1.3p+0) { // 19/16 > v > 11/16
        a = v - 1.0;
        b = 1.0 + v;
        // (chi + clo) = arctan(1.)
        chi = 0x1.921fb54442d18p-1;
        clo = 0x1.1a62633145c06p-55;
    } else if (v <= 0x1.38p+1) { // 39/16 > v > 19/16
        a = v - 1.5;
        b = MATH_MAD(v, 1.5, 1.0);
        // (chi + clo) = arctan(1.5)
        chi = 0x1.f730bd281f69bp-1;
        clo = 0x1.007887af0cbbcp-56;
    } else { // 2^56 > v > 39/16
        a = -1.0;
        b = v;
        // (chi+clo) = arctan(Inf)
        chi = 0x1.921fb54442d18p+0;
        clo = 0x1.1a62633145c06p-54;
    }
#else
    // 2^56 > v > 39/16
    double a = -1.0;
    double b = v;
    // (chi + clo) = arctan(infinity)
    double chi = 0x1.921fb54442d18p+0;
    double clo = 0x1.1a62633145c06p-54;

    double ta = v - 1.5;
    double tb = MATH_MAD(v, 1.5, 1.0);
    bool l = v <= 0x1.38p+1; // 39/16 > v > 19/16
    a = l ? ta : a;
    b = l ? tb : b;
    // (chi + clo) = arctan(1.5)
    chi = l ? 0x1.f730bd281f69bp-1 : chi;
    clo = l ? 0x1.007887af0cbbcp-56 : clo;

    ta = v - 1.0;
    tb = 1.0 + v;
    l = v <= 0x1.3p+0; // 19/16 > v > 11/16
    a = l ? ta : a;
    b = l ? tb : b;
    // (chi + clo) = arctan(1.)
    chi = l ? 0x1.921fb54442d18p-1 : chi;
    clo = l ? 0x1.1a62633145c06p-55 : clo;

    ta = MATH_MAD(v, 2.0, -1.0);
    tb = 2.0 + v;
    l = v <= 0x1.6p-1; // 11/16 > v > 7/16
    a = l ? ta : a;
    b = l ? tb : b;
    // (chi + clo) = arctan(0.5)
    chi = l ? 0x1.dac670561bb4fp-2 : chi;
    clo = l ? 0x1.a2b7f222f65e0p-56 : clo;

    l = v <= 0x1.cp-2; // v < 7/16
    a = l ? v : a;
    b = l ? 1.0 : b;;
    chi = l ? 0.0 : chi;
    clo = l ? 0.0 : clo;
#endif

    // Core approximation: Remez(4,4) on [-7/16,7/16]
    double r = MATH_FAST_DIV(a,  b);
    double s = r * r;
    double qn = MATH_MAD(s,
                    MATH_MAD(s,
                        MATH_MAD(s,
                            MATH_MAD(s, 0x1.2a75ce41b9f87p-13, 0x1.f2d2116f053f2p-6),
                            0x1.c3de43db425c0p-3),
                        0x1.ca6be4c993b3cp-2),
                    0x1.12bcb0a9169f3p-2);

    double qd = MATH_MAD(s,
	            MATH_MAD(s,
			MATH_MAD(s,
			    MATH_MAD(s, 0x1.3f197f1e85ed9p-5, 0x1.b2cb05bf9beffp-2),
                            0x1.699c644c48d2ep+0),
                        0x1.d372a17cdf5a0p+0),
                    0x1.9c1b08fda1eecp-1);

    double q = r * s * MATH_FAST_DIV(qn, qd);
    r = chi - ((q - clo) - r);

    double z;

    if (!FINITE_ONLY_OPT()) {
        z = BUILTIN_CLASS_F64(x, CLASS_QNAN|CLASS_SNAN) ? x : piby2;
    } else {
	z = piby2;
    }

    z = v <= 0x1.0p+56 ? r : z;
    z = v < 0x1.0p-26 ? v : z;
    return BUILTIN_COPYSIGN_F64(z, x);
}

