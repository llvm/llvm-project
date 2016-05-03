
#include "mathF.h"

PUREATTR INLINEATTR float
MATH_MANGLE(rsqrt)(float x)
{
    if (DAZ_OPT() & AMD_OPT()) {
        return BUILTIN_RSQRT_F32(x);
    } else if (AMD_OPT()) {
        int s = x < 0x1.0p-100f;
        x *= s ? 0x1.0p+100f : 1.0f;
        x = BUILTIN_RSQRT_F32(x);
        x *= s ? 0x1.0p+50f : 1.0f;
        return x;
    } else {
        USE_TABLE(float, p_tbl, M32_RSQRT);
        float y = x * (x < 0x1.0p-100f ? 0x1.0p+100f : 1.0f);
        int e = (as_int(y) >> 23) - 127;
        int i = ((e & 1) << 5) + ((as_int(y) >> 18) & 0x1f);
        float r = p_tbl[i] * as_float((127 - (e >> 1)) << 23);
        r = r * MATH_MAD(-y*r*0.5f, r, 1.5f);
        r = r * MATH_MAD(-y*r*0.5f, r, 1.5f);
        r = r * MATH_MAD(-y*r*0.5f, r, 1.5f);
        r = r * (x < 0x1.0p-100f ? 0x1.0p+50f : 1.0f);
        if (!FINITE_ONLY_OPT()) {
            int ix = as_int(x);
            r = ix == PINFBITPATT_SP32 ? 0.0 : r;
            r = ix > PINFBITPATT_SP32 | ix < 0  ? as_float(QNANBITPATT_SP32) : r;
            float inf = as_float((ix & SIGNBIT_SP32) | PINFBITPATT_SP32);
            r = x == 0.0f ? inf : r;
        }
        return r;
    }
}

