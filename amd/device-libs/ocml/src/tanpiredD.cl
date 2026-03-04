/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathD.h"
#include "trigpiredD.h"

CONSTATTR double
MATH_PRIVATE(tanpired)(double x, int i)
{
    double s = x * x;
    double t = MATH_MAD(s, MATH_MAD(s, MATH_MAD(s, MATH_MAD(s, 
               MATH_MAD(s, MATH_MAD(s, MATH_MAD(s, MATH_MAD(s, 
               MATH_MAD(s, MATH_MAD(s, MATH_MAD(s, MATH_MAD(s, 
               MATH_MAD(s, 
                   0x1.3fad0a71ea6d1p+32, -0x1.11a76ac97377bp+30), 0x1.ba2bcaca6da1bp+27), -0x1.79e8e2d7aaf57p+22),
                   0x1.c1c1102e46eccp+21), 0x1.31291bbcb5588p+19), 0x1.486b2d6bb3db2p+17), 0x1.45be1b46ff156p+15),
                   0x1.45f61b419c746p+13), 0x1.45f311045a4ffp+11), 0x1.45f4739a998c7p+9), 0x1.45fff9b243050p+7),
                   0x1.466bc6775cf74p+5), 0x1.4abbce625be8bp+3);
    t = x * s * t;
    t = MATH_MAD(x, 0x1.921fb54442d18p+1, t);

    double tr = -MATH_RCP(t);

    return i ? tr : t;
}

