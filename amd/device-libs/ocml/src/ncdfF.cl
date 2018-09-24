/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathF.h"

CONSTATTR float
MATH_MANGLE(ncdf)(float x)
{
    float ret;

    // cut at -0x1.5956b8p-1f

    if (x > -0x1.5956b8p-1f) {
        if (x < 1.0f) {
            float t = x*x;
            ret = MATH_MAD(t, MATH_MAD(t, MATH_MAD(t, MATH_MAD(t, 
                  MATH_MAD(t, MATH_MAD(t, 
                      0x1.20379ep-21f, -0x1.3727aep-17f), 0x1.e3af2ep-14f), -0x1.373d8cp-10f),
                      0x1.46d034p-7f), -0x1.105838p-4f), 0x1.988454p-2f);
            ret = MATH_MAD(x, ret, 0.5f);
        } else if (x < 2.5f) {
            float t = x - 1.0f;
            ret = MATH_MAD(t, MATH_MAD(t, MATH_MAD(t, MATH_MAD(t, 
                  MATH_MAD(t, MATH_MAD(t, MATH_MAD(t, MATH_MAD(t, 
                      -0x1.53eaecp-13f, 0x1.3458b4p-10f), -0x1.306adcp-9f), -0x1.01ae44p-8f),
                      0x1.4a7e5ep-6f), -0x1.fe4012p-17f), -0x1.ef8a62p-4f), 0x1.ef8e32p-3f),
                      0x1.aec4bep-1f);
        } else if (x < 4.0f) {
            float t = x - 2.5f;
            ret = MATH_MAD(t, MATH_MAD(t, MATH_MAD(t, MATH_MAD(t, 
                  MATH_MAD(t, MATH_MAD(t, 
                      -0x1.4ca664p-13f, 0x1.990fd2p-10f), -0x1.b0d706p-8f), 0x1.ffa500p-7f),
                      -0x1.67e84cp-6f), 0x1.1f419cp-6f), 0x1.fcd214p-1f);
        } else if (x < 5.296875f) {
            float t = x - 4.0f;
            ret = MATH_MAD(t, MATH_MAD(t, MATH_MAD(t, 
                      0x1.eae60ap-10f, 0x1.9b6438p-9f), -0x1.1b57a8p-3f), 0x1.0bf538p-1f);
            ret = ret * ret;
            ret = ret * ret;
            ret = ret * ret;
            ret = MATH_MAD(-ret, ret, 1.0f);
        } else {
            ret = 1.0f;
        }
    } else {
        if (x > -1.5f) {
             float t = -1.5f - x;
             ret = MATH_MAD(t, MATH_MAD(t, MATH_MAD(t, MATH_MAD(t, 
                   MATH_MAD(t, MATH_MAD(t, MATH_MAD(t, 
                       -0x1.a29ef2p-11f, -0x1.a25e42p-11f), 0x1.7eaaaap-8f), -0x1.8d95e2p-8f),
                       -0x1.ba093ap-6f), 0x1.8de146p-4f), -0x1.094082p-3f), 0x1.11a46ep-4f);
        } else if (x > -2.5f) {
            float t = -2.5f - x;
            ret = MATH_MAD(t, MATH_MAD(t, MATH_MAD(t, MATH_MAD(t, 
                  MATH_MAD(t, MATH_MAD(t, MATH_MAD(t, MATH_MAD(t, 
                      0x1.aef5d0p-14f, 0x1.0b8148p-11f), -0x1.232788p-12f), -0x1.1afa4cp-11f),
                      0x1.877322p-8f), -0x1.f65b2ep-7f), 0x1.66fd08p-6f), -0x1.1f2ef4p-6f),
                      0x1.96f4e6p-8f);
        } else if (x > -3.25f) {
            float t = -3.25f - x;
            ret = MATH_MAD(t, MATH_MAD(t, MATH_MAD(t, MATH_MAD(t, 
                  MATH_MAD(t, MATH_MAD(t, MATH_MAD(t, MATH_MAD(t, 
                      -0x1.8963dep-15f, -0x1.2e81a4p-17f), 0x1.7477b2p-13f), -0x1.c8841ap-11f),
                      0x1.1036c6p-9f), -0x1.a7e084p-9f), 0x1.b02b86p-9f), -0x1.09f390p-9f),
                      0x1.2e86fep-11f);
        } else if (x > -14.125f) {
            float t = MATH_FAST_RCP(x * x);

            if (x > -5.0f)
                ret = MATH_MAD(t, MATH_MAD(t, MATH_MAD(t, MATH_MAD(t, 
                      MATH_MAD(t, MATH_MAD(t, 
                          0x1.f9b114p+7f, -0x1.32f4b4p+7f), 0x1.723550p+5f), -0x1.4b98dcp+3f),
                          0x1.3821cep+1f), -0x1.ff6d7cp-1f), -0x1.4023a6p-8f);
            else
                ret = MATH_MAD(t, MATH_MAD(t, MATH_MAD(t, MATH_MAD(t, 
                      MATH_MAD(t, MATH_MAD(t, 
                          0x1.f31adep+10f, -0x1.030fd6p+9f), 0x1.41d2c6p+6f), -0x1.86b97ap+3f),
                          0x1.3fdb64p+1f), -0x1.ffff50p-1f), -0x1.3f8e6cp-8f);

            float xh = AS_FLOAT(AS_INT(x) & 0xffffe000);
            ret = MATH_FAST_DIV(MATH_MANGLE(exp)(MATH_MAD(x - xh,  -0.5f*(x + xh), ret)), -x) *
                  MATH_MANGLE(exp)(MATH_MAD(xh, -0.5f*xh, -0.9140625f));
        } else {
            ret = BUILTIN_CLASS_F32(x, CLASS_QNAN|CLASS_SNAN) ? x : 0.0f;
        }
    }

    return ret;
}

