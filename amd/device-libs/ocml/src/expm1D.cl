/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathD.h"

PUREATTR double
MATH_MANGLE(expm1)(double x)
{
    USE_TABLE(double2, p_tbl, M64_EXP_EP);

    const double max_expm1_arg = 709.8;
    const double min_expm1_arg = -37.42994775023704;
    const double log_OnePlus_OneByFour = 0.22314355131420976;   //0x3FCC8FF7C79A9A22 = log(1+1/4)
    const double log_OneMinus_OneByFour = -0.28768207245178096; //0xBFD269621134DB93 = log(1-1/4)
    const double sixtyfour_by_lnof2 = 92.33248261689366;        //0x40571547652b82fe
    const double lnof2_by_64_head = 0.010830424696223417;       //0x3f862e42fefa0000
    const double lnof2_by_64_tail = 2.5728046223276688e-14;     //0x3d1cf79abc9e3b39

    double z;

    if (x > log_OneMinus_OneByFour & x < log_OnePlus_OneByFour) {
        double u = AS_DOUBLE(AS_ULONG(x) & 0xffffffffff000000UL);
        double v = x - u;
        double y = u * u * 0.5;
        double t = v * (x + u) * 0.5;

        double q = MATH_MAD(x,
	               MATH_MAD(x,
		           MATH_MAD(x,
			       MATH_MAD(x,
			           MATH_MAD(x,
				       MATH_MAD(x,
				           MATH_MAD(x,
					       MATH_MAD(x,2.4360682937111612e-8, 2.7582184028154370e-7),
					       2.7558212415361945e-6),
				           2.4801576918453420e-5),
				       1.9841269447671544e-4),
			           1.3888888890687830e-3),
			       8.3333333334012270e-3),
		           4.1666666666665560e-2),
		       1.6666666666666632e-1);
        q *= x * x * x;

        double z1g = (u + y) + (q + (v + t));
        double z1 = x + (y + (q + t));
        z = y >= 0x1.0p-7 ? z1g : z1;
    } else {
        int n = (int)(x * sixtyfour_by_lnof2);
        int j = n & 0x3f;
        int m = n >> 6;

        double2 tv = p_tbl[j];
        double f1 = tv.s0;
        double f2 = tv.s1;
        double f = f1 + f2;

        double dn = -n;
        double r = MATH_MAD(dn, lnof2_by_64_tail, MATH_MAD(dn, lnof2_by_64_head, x));

        double q = MATH_MAD(r,
	               MATH_MAD(r,
		           MATH_MAD(r,
		               MATH_MAD(r, 1.38889490863777199667e-03, 8.33336798434219616221e-03),
		               4.16666666662260795726e-02),
		           1.66666666665260878863e-01),
	                5.00000000000000008883e-01);
        q = MATH_MAD(r*r, q, r);

        double twopm = AS_DOUBLE((long)(EXPBIAS_DP64 + m) << EXPSHIFTBITS_DP64);
        double twopmm = AS_DOUBLE((long)(EXPBIAS_DP64 - m) << EXPSHIFTBITS_DP64);

        // Computations for m > 52, including where result is close to Inf
        ulong uval = AS_ULONG(0x1.0p+1023 * (f1 + (f * q + (f2))));
        int e = (int)(uval >> EXPSHIFTBITS_DP64) + 1;

        double zme1024 = AS_DOUBLE(((long)e << EXPSHIFTBITS_DP64) | (uval & MANTBITS_DP64));
        zme1024 = e == 2047 ? AS_DOUBLE(PINFBITPATT_DP64) : zme1024;

        double zmg52 = twopm * (f1 + MATH_MAD(f, q, f2 - twopmm));
        zmg52 = m == 1024 ? zme1024 : zmg52;

        // For m < 53
        double zml53 = twopm * ((f1 - twopmm) + MATH_MAD(f1, q, f2*(1.0 + q)));

        // For m < -7
        double zmln7 = MATH_MAD(twopm,  f1 + MATH_MAD(f, q, f2), -1.0);

        z = m < 53 ? zml53 : zmg52;
        z = m < -7 ? zmln7 : z;
    }

    if (!FINITE_ONLY_OPT()) {
        z = x > max_expm1_arg ? AS_DOUBLE(PINFBITPATT_DP64) : z;
    }

    z = x < min_expm1_arg ? -1.0 : z;

    return z;
}

