
#include "mathF.h"

CONSTATTR INLINEATTR float
MATH_MANGLE(atan)(float x)
{
    const float piby2 = 0x1.921fb6p+0f;

    uint ux = as_uint(x);
    uint aux = ux & EXSIGNBIT_SP32;
    uint sx = ux ^ aux;
    float v = as_float(aux);
    float ret;

    // 2^26 <= |x| <= Inf => atan(x) is close to piby2
    if (!FINITE_ONLY_OPT()) {
        ret = aux <= PINFBITPATT_SP32  ? piby2 : x;
    } else {
        ret = piby2;
    }

    // Reduce arguments 2^-13 <= |x| < 2^26

    // 39/16 <= x < 2^26
    float a = -1.0f;
    float b = v;
    float c = piby2; // atan(infinity)

    // 19/16 <= x < 39/16
    bool l = aux < 0x401c0000;
    float ta = v - 1.5f;
    float tb = MATH_MAD(v, 1.5f, 1.0f);
    a = l ? ta : a;
    b = l ? tb : b;
    c = l ? 0x1.f730bep-1f : c; // atan(1.5)

    // 11/16 <= x < 19/16
    l = aux < 0x3f980000U;
    ta = v - 1.0f;
    tb = 1.0f + v;
    a = l ? ta : a;
    b = l ? tb : b;
    c = l ? 0x1.921fb6p-1f : c; // atan(1)

    // 7/16 <= x < 11/16
    l = aux < 0x3f300000;
    ta = MATH_MAD(v, 2.0f, -1.0f);
    tb = 2.0f + v;
    a = l ? ta : a;
    b = l ? tb : b;
    c = l ? 0x1.dac670p-2f : c; // atan(0.5)

    // 2^-13 <= x < 7/16
    l = aux < 0x3ee00000;
    a = l ? v : a;
    b = l ? 1.0f : b;
    c = l ? 0.0f : c;

    // Core approximation: Remez(2,2) on [-7/16,7/16]

    x = MATH_FAST_DIV(a, b);
    float x2 = x * x;

    float p = MATH_MAD(x2, MATH_MAD(x2, -0x1.3476a8p-8f, -0x1.89e174p-3f), -0x1.2fa532p-2f);
    float q = MATH_MAD(x2, MATH_MAD(x2, 0x1.327e3ep-2f, 0x1.1c587ap+0f), 0x1.c777cap-1f);

    float r = c + MATH_MAD(p*x2*x, MATH_FAST_RCP(q), x);

    ret = aux < 0x4c800000 ? r : ret;
    ret = aux < 0x39000000 ? v : ret;

    return as_float(sx | as_uint(ret));
}

