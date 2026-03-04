/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathD.h"

CONSTATTR double
MATH_PRIVATE(atanred)(double v)
{
    double t = v * v;
    double z = MATH_MAD(t, MATH_MAD(t, MATH_MAD(t, MATH_MAD(t, 
               MATH_MAD(t, MATH_MAD(t, MATH_MAD(t, MATH_MAD(t, 
               MATH_MAD(t, MATH_MAD(t, MATH_MAD(t, MATH_MAD(t, 
               MATH_MAD(t, MATH_MAD(t, MATH_MAD(t, MATH_MAD(t, 
               MATH_MAD(t, MATH_MAD(t, MATH_MAD(t, 
                   0x1.ba404b5e68a13p-17, -0x1.3e260bd3237f4p-13), 0x1.b2bb069efb384p-11), -0x1.7952daf56de9bp-9),
                   0x1.d6d43a595c56fp-8), -0x1.c6ea4a57d9582p-7), 0x1.67e295f08b19fp-6), -0x1.e9ae6fc27006ap-6),
                   0x1.2c15b5711927ap-5), -0x1.59976e82d3ff0p-5), 0x1.82d5d6ef28734p-5), -0x1.ae5ce6a214619p-5),
                   0x1.e1bb48427b883p-5), -0x1.110e48b207f05p-4), 0x1.3b13657b87036p-4), -0x1.745d119378e4fp-4),
                   0x1.c71c717e1913cp-4), -0x1.2492492376b7dp-3), 0x1.99999999952ccp-3), -0x1.5555555555523p-2);
    z = MATH_MAD(v, t*z, v);
    return z;
}

