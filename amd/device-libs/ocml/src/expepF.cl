/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathF.h"

#define FLOAT_SPECIALIZATION
#include "ep.h"

INLINEATTR CONSTATTR float
MATH_PRIVATE(expep)(float2 x)
{
    float fn = BUILTIN_RINT_F32(x.hi * 0x1.715476p+0f);
    float2 t = fsub(fsub(sub(x, fn*0x1.62e400p-1f), fn*0x1.7f7800p-20f), fn*0x1.473de6p-34f);

    float th = t.hi;
    float p = MATH_MAD(th, MATH_MAD(th, MATH_MAD(th, MATH_MAD(th, 
                  0x1.6850e4p-10f, 0x1.123bccp-7f), 0x1.555b98p-5f), 0x1.55548ep-3f),
                  0x1.fffff8p-2f);

    float2 r = fadd(t, mul(sqr(t), p));
    float z = 1.0 + r.hi;

    z = BUILTIN_FLDEXP_F32(z, (int)fn);

    z = x.hi > 89.0f ? AS_FLOAT(PINFBITPATT_SP32) : z;
    z = x.hi < -104.0f ? 0.0f : z;

    return z;
}

