/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathD.h"

CONSTATTR double
MATH_MANGLE(tgamma)(double x)
{
    const double pi = 3.14159265358979323846;


    double ax = BUILTIN_ABS_F64(x);
    double ret;

    if (ax > 0x1.0p-11) {
        // For x < 4, push to [1-3] range  using gamma(x) = gamma(x+1) / x
        // For 4.5 < x < 6.5, push above 6.5
        // [4,4.5) left alone
        double nterm = 1.0;
        double dterm = 1.0;
        double z = ax;
        if (ax < 4.5) {
            if (ax < 1.0) {
                dterm = z;
                z += 1.0;
            } else if (ax < 3.0) {
                ; // do nothing
            } else if (ax < 4.0) {
                z -= 1.0;
                nterm = z;
            }
        } else if (ax < 5.5) {
            dterm = MATH_MAD(z,z,z);
            z += 2.0;
        } else if (ax < 6.5) {
            dterm = z;
            z += 1.0;
        }

        double negadj = 1.0;
        if (x < 0.0) {
            negadj = -x * MATH_MANGLE(sinpi)(x);
        }

        double etonegz = MATH_MANGLE(exp)(-z);

        if (z < 4.5) {
            const double rn0 =     297.312130630940277;
            const double rn1 =   16926.1409177878806;
            const double rn2 =  131675.407800922036;
            const double rn3 =  344586.743316038732;
            const double rn4 =  440619.954224349898;
            const double rn5 =  275507.567385621460;
            const double rn6 =   84657.9644812230335;

            const double rd0 =       1.00000000000000000;
            const double rd1 =     -13.3400904528209096;
            const double rd2 =    3270.94389286527964;
            const double rd3 =   41972.5365974090031;
            const double rd4 =  123293.896672792281;
            const double rd5 =  166739.899991898533;
            const double rd6 =  107097.146935059144;
            const double rd7 =   33773.6414083704053;

            double num = MATH_MAD(z,
                             MATH_MAD(z,
                                 MATH_MAD(z,
                                     MATH_MAD(z,
                                         MATH_MAD(z,
                                             MATH_MAD(z, rn6, rn5),
                                             rn4),
                                         rn3),
                                     rn2),
                                 rn1),
                             rn0) * nterm;

            double den = MATH_MAD(z,
                            MATH_MAD(z,
                                MATH_MAD(z,
                                    MATH_MAD(z,
                                        MATH_MAD(z,
                                            MATH_MAD(z,
                                                MATH_MAD(z, rd7, rd6),
                                                rd5),
                                            rd4),
                                        rd3),
                                    rd2),
                                rd1),
                            rd0) * dterm;

            double zpow = MATH_MANGLE(powr)(z, z+0.5);

            if (x >= 0.0) {
                ret = etonegz * zpow * MATH_DIV(num,den);
            } else {
                ret = MATH_DIV(den*pi, negadj*etonegz*zpow*num);
                ret = BUILTIN_FRACTION_F64(x) == 0.0 ? QNAN_F64 : ret;
            }
        } else {
            const double c0  =  2.5066282746310007;
            const double c1  =  0.20888568955258338;
            const double c2  =  0.008703570398024307;
            const double c3  = -0.0067210904740298821;
            const double c4  = -0.00057520123811017124;
            const double c5  =  0.0019652948815832029;
            const double c6  =  0.00017478252120455912;
            const double c7  = -0.0014843411351582762;
            const double c8  = -0.00012963757321125544;
            const double c9  =  0.0021043112297532062;
            const double c10 =  0.00018059994565555043;
            const double c11 = -0.0047987856705463457;
            const double c12 = -0.0004073678593815252;
            const double c13 =  0.01605085033194459500;
            const double c14 =  0.0013539922801590941;
            const double c15 = -0.074015421268427375;
            const double c16 = -0.0062208086788087787;
            const double c17 =  0.45004033385625097;

            double rz = MATH_RCP(z);

            double poly = MATH_MAD(rz, MATH_MAD(rz, MATH_MAD(rz, MATH_MAD(rz, MATH_MAD(rz,
                          MATH_MAD(rz, MATH_MAD(rz, MATH_MAD(rz, MATH_MAD(rz, MATH_MAD(rz,
                          MATH_MAD(rz, MATH_MAD(rz, MATH_MAD(rz, MATH_MAD(rz, MATH_MAD(rz,
                          MATH_MAD(rz, MATH_MAD(rz,
                          c17, c16), c15), c14), c13), c12), c11), c10), c9), c8),
                               c7), c6), c5), c4), c3), c2), c1), c0);

            double zpow = MATH_MANGLE(powr)(z, MATH_MAD(0.5, z, -0.25));
            if (x >= 0.0) {
                ret = MATH_DIV(etonegz*zpow*zpow*poly, dterm);
                ret = x > 0x1.573fae561f647p+7 ? PINF_F64 : ret;
            } else if (x < 0.0) {
                if (x >= -170.5) {
                    ret = MATH_DIV(pi*dterm, etonegz*zpow*zpow*poly*negadj);
                } else if (x >= -184.0) {
                    ret = MATH_DIV(MATH_DIV(pi*dterm, etonegz*zpow*poly), zpow*negadj);
                } else {
                    ret = BUILTIN_COPYSIGN_F64(0.0, negadj);
                }
                ret = BUILTIN_FRACTION_F64(x) == 0.0 ? QNAN_F64 : ret;
            } else {
                ret = x;
            }
        }
    } else {
        const double c0 = -0x1.2788cfc6fb619p-1;
        const double c1 =  0x1.fa658c23b1578p-1;
        const double c2 = -0x1.d0a118f324b63p-1;
        const double c3 =  0x1.f6a51055096b5p-1;

        ret = MATH_MAD(x, MATH_MAD(x, MATH_MAD(x, c3, c2), c1), c0) + MATH_RCP(x);
    }

    return ret;
}

