/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathD.h"

INLINEATTR double
MATH_PRIVATE(sincospired)(double x, __private double *cp)
{

    double t = x * x;

    double sx = MATH_MAD(t, MATH_MAD(t, MATH_MAD(t, MATH_MAD(t, 
                MATH_MAD(t,
                    0x1.e357ef99eb0bbp-12, -0x1.e2fe76fdffd2bp-8), 0x1.50782d5f14825p-4), -0x1.32d2ccdfe9424p-1),
                    0x1.466bc67754fffp+1), -0x1.4abbce625be09p+2);
    sx = x * t * sx;
    sx = MATH_MAD(x, 0x1.921fb54442d18p+1, sx);

    double cx = MATH_MAD(t, MATH_MAD(t, MATH_MAD(t, MATH_MAD(t, 
                MATH_MAD(t, MATH_MAD(t, 
                    -0x1.b167302e21c33p-14, 0x1.f9c89ca1d4f33p-10), -0x1.a6d1e7294bff9p-6), 0x1.e1f5067b90b37p-3),
                    -0x1.55d3c7e3c325bp+0), 0x1.03c1f081b5a67p+2), -0x1.3bd3cc9be45dep+2);
    cx = MATH_MAD(t, cx, 1.0);

    *cp = cx;
    return sx;
}

