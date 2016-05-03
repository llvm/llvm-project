
#include "mathH.h"
#include "trigredH.h"

CONSTATTR INLINEATTR half
MATH_PRIVATE(tanred)(half hx, int regn)
{
    const float a0 = 0.385296071263995406715129f;
    const float a1 = -0.0172032480471481694693109f;
    const float b0 = 1.15588821434688393452299f;
    const float b1 = -0.51396505478854532132342f;
    const float b2 = 0.01844239256901656082986661f;

    // Unfortunately, we're over 2ulp with this computation in half precision
    float x = (float)hx;
    float r = x * x;
    half ret;

    if (AMD_OPT()) {
        float a = BUILTIN_MAD_F32(r, a1, a0);
        float b = BUILTIN_MAD_F32(r, BUILTIN_MAD_F32(r, b2, b1), b0);
        float q = a * BUILTIN_RCP_F32(b);
        float t = BUILTIN_MAD_F32(x*r, q, x);
        float tr = -BUILTIN_RCP_F32(t);
        ret = (half)(regn & 1 ? tr : t);
    } else {
        float a = r*a1 + a0;
        float b = (r*b2 + b1)*r + b0;
        float q = a / b;
        float t = x*r*q + x;
        float tr = -1.0f / t;
        ret = (half)(regn & 1 ? tr : t);
    }

    return ret;
}

