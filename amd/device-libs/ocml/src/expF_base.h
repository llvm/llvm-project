/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathF.h"

//    Algorithm:
// 
//    e^x = 2^(x/ln(2)) = 2^(x*(64/ln(2))/64)
// 
//    x*(64/ln(2)) = n + f, |f| <= 0.5, n is integer
//    n = 64*m + j,   0 <= j < 64
// 
//    e^x = 2^((64*m + j + f)/64)
//        = (2^m) * (2^(j/64)) * 2^(f/64)
//        = (2^m) * (2^(j/64)) * e^(f*(ln(2)/64))
// 
//    f = x*(64/ln(2)) - n
//    r = f*(ln(2)/64) = x - n*(ln(2)/64)
// 
//    e^x = (2^m) * (2^(j/64)) * e^r
// 
//    (2^(j/64)) is precomputed
// 
//    e^r = 1 + r + (r^2)/2! + (r^3)/3! + (r^4)/4! + (r^5)/5!
//    e^r = 1 + q
// 
//    q = r + (r^2)/2! + (r^3)/3! + (r^4)/4! + (r^5)/5!
// 
//    e^x = (2^m) * ( (2^(j/64)) + q*(2^(j/64)) ) 

CONSTATTR float
MATH_MANGLE(exp10)(float x)
{
    if (DAZ_OPT()) {
        if (UNSAFE_MATH_OPT()) {
            return BUILTIN_AMDGPU_EXP2_F32(x * 0x1.a92000p+1f) * BUILTIN_AMDGPU_EXP2_F32(x * 0x1.4f0978p-11f);
        } else {
            float ph, pl;

            if (HAVE_FAST_FMA32()) {
                const float c = 0x1.a934f0p+1f;
                const float cc = 0x1.2f346ep-24f;
                ph = x * c;
                pl = BUILTIN_FMA_F32(x, cc, BUILTIN_FMA_F32(x, c, -ph));
            } else {
                const float ch = 0x1.a92000p+1f;
                const float cl = 0x1.4f0978p-11f;
                float xh = AS_FLOAT(AS_INT(x) & 0xfffff000);
                float xl = x - xh;
                ph = xh * ch;
                pl = MATH_MAD(xh, cl, MATH_MAD(xl, ch, xl*cl));
            }

            float e = BUILTIN_RINT_F32(ph);
            float a = ph - e + pl;
            float r = BUILTIN_FLDEXP_F32(BUILTIN_AMDGPU_EXP2_F32(a), (int)e);

            r = x < -0x1.2f7030p+5f ? 0.0f : r;
            r = x > 0x1.344136p+5f ? PINF_F32 : r;
            return r;
        }
    } else {
        if (UNSAFE_MATH_OPT()) {
            bool s = x < -0x1.2f7030p+5f;
            x += s ? 0x1.0p+5f : 0.0f;
            return BUILTIN_AMDGPU_EXP2_F32(x * 0x1.a92000p+1f) *
                   BUILTIN_AMDGPU_EXP2_F32(x * 0x1.4f0978p-11f) *
                   (s ? 0x1.9f623ep-107f : 1.0f);
        } else {
            float ph, pl;

            if (HAVE_FAST_FMA32()) {
                const float c = 0x1.a934f0p+1f;
                const float cc = 0x1.2f346ep-24f;
                ph = x * c;
                pl = BUILTIN_FMA_F32(x, cc, BUILTIN_FMA_F32(x, c, -ph));
            } else {
                const float ch = 0x1.a92000p+1f;
                const float cl = 0x1.4f0978p-11f;
                float xh = AS_FLOAT(AS_INT(x) & 0xfffff000);
                float xl = x - xh;
                ph = xh * ch;
                pl = MATH_MAD(xh, cl, MATH_MAD(xl, ch, xl*cl));
            }

            float e = BUILTIN_RINT_F32(ph);
            float a = ph - e + pl;
            float r = BUILTIN_FLDEXP_F32(BUILTIN_AMDGPU_EXP2_F32(a), (int)e);

            r = x < -0x1.66d3e8p+5f ? 0.0f : r;
            r = x > 0x1.344136p+5f ? PINF_F32 : r;
            return r;
        }
    }
}

