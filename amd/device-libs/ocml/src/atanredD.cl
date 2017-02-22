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
               MATH_MAD(t, MATH_MAD(t, MATH_MAD(t, MATH_MAD(t, 
                   -0x1.9435e2f8ccf2ep-17, 0x1.238caa835c8a6p-13), -0x1.8f6beee6228dcp-11), 0x1.5b88ab0ab8f42p-9),
                   -0x1.b29a028bf93e4p-8), 0x1.a499d84d6f67ep-7), -0x1.4cfd320285122p-6), 0x1.c4d30b9f0bbf9p-6),
                   -0x1.14d953e4d5be8p-5), 0x1.3d6e10090f255p-5), -0x1.60fe035db81cap-5), 0x1.854ff19df1d43p-5),
                   -0x1.af01fe33c55fcp-5), 0x1.e1dc44c2de6d1p-5), -0x1.1110c3667d566p-4), 0x1.3b13ab43bf0ecp-4),
                   -0x1.745d16f715fedp-4), 0x1.c71c71c49d4b4p-4), -0x1.249249248ce45p-3), 0x1.9999999999904p-3),
                   -0x1.5555555555555p-2);
    z = MATH_MAD(v, t*z, v);
    return z;
}

