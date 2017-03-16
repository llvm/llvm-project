/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathD.h"

CONSTATTR double
MATH_PRIVATE(atanpired)(double v)
{
    double t = v * v;
    double z = MATH_MAD(t, MATH_MAD(t, MATH_MAD(t, MATH_MAD(t, 
               MATH_MAD(t, MATH_MAD(t, MATH_MAD(t, MATH_MAD(t, 
               MATH_MAD(t, MATH_MAD(t, MATH_MAD(t, MATH_MAD(t, 
               MATH_MAD(t, MATH_MAD(t, MATH_MAD(t, MATH_MAD(t, 
               MATH_MAD(t, MATH_MAD(t, MATH_MAD(t, MATH_MAD(t, 
                   0x1.39e58b43320d2p-18, -0x1.be9e52f5df14fp-15), 0x1.2d7a6cad8e9dbp-12), -0x1.024ebcc10f8a6p-10),
                   0x1.3df92946a87d8p-9), -0x1.2f04271b6cd94p-8), 0x1.d91b9a6908690p-8), -0x1.3e1c18f5ea692p-7),
                   0x1.8253e53662be6p-7), -0x1.ba3db7e462112p-7), 0x1.ed7188505388cp-7), -0x1.121f707a5851bp-6),
                   0x1.32b737d7f904ap-6), -0x1.5bac13378ea68p-6), 0x1.912af944c4411p-6), -0x1.da1babd44fccfp-6),
                   0x1.21bb945aacd29p-5), -0x1.7483758f7040fp-5), 0x1.04c26be3b5934p-4), -0x1.b2995e7b7b74dp-4),
                   0x1.45f306dc9c883p-2);
    return v * z;
}

