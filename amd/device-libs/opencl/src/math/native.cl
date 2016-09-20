/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "irif.h"

// Value of log2(10)
#define M_LOG2_10_F 0x1.a934f0p+1f

// Value of 1 / log2(10)
#define M_RLOG2_10_F 0x1.344136p-2f

// Value of 1 / M_LOG2E_F = 1 / log2(e)
#define M_RLOG2_E_F 0x1.62e430p-1f

#define ATTR __attribute__((always_inline, overloadable, const))

#if !defined USE_CLP
#define LISTU2(F) F(x.s0), F(x.s1)
#define LISTU3(F) F(x.s0), F(x.s1), F(x.s2)
#define LISTU4(F) LISTU2(F), F(x.s2), F(x.s3)
#define LISTU8(F) LISTU4(F), F(x.s4), F(x.s5), F(x.s6), F(x.s7)
#define LISTU16(F) LISTU8(F), F(x.s8), F(x.s9), F(x.sa), F(x.sb), \
                                F(x.sc), F(x.sd), F(x.se), F(x.sf)

#define EXPUN(N,F) \
ATTR float##N \
F(float##N x) \
{ \
    return (float##N) ( LISTU##N(F) ); \
}

#define EXPU(F) \
    EXPUN(16,F) \
    EXPUN(8,F) \
    EXPUN(4,F) \
    EXPUN(3,F) \
    EXPUN(2,F)

#define LISTB2(F) F(x.s0,y.s0), F(x.s1,y.s1)
#define LISTB3(F) F(x.s0,y.s0), F(x.s1,y.s1), F(x.s2,y.s2)
#define LISTB4(F) LISTB2(F), F(x.s2,y.s2), F(x.s3,y.s3)
#define LISTB8(F) LISTB4(F), F(x.s4,y.s4), F(x.s5,y.s5), F(x.s6,y.s6), F(x.s7,y.s7)
#define LISTB16(F) LISTB8(F), F(x.s8,y.s8), F(x.s9,y.s9), F(x.sa,y.sa), F(x.sb,y.sb), \
                              F(x.sc,y.sc), F(x.sd,y.sd), F(x.se,y.se), F(x.sf,y.sf)

#define EXPBN(N,F) \
ATTR float##N \
F(float##N x, float##N y) \
{ \
    return (float##N) ( LISTB##N(F) ); \
}

#define EXPB(F) \
    EXPBN(16,F) \
    EXPBN(8,F) \
    EXPBN(4,F) \
    EXPBN(3,F) \
    EXPBN(2,F)


EXPB(native_divide)
EXPB(native_powr)
EXPU(native_tan)
EXPU(native_cos)
EXPU(native_exp)
EXPU(native_exp2)
EXPU(native_exp10)
EXPU(native_log)
EXPU(native_log2)
EXPU(native_log10)
EXPU(native_recip)
EXPU(native_rsqrt)
EXPU(native_sin)
EXPU(native_sqrt)
#endif // !USE_CLP

ATTR float
native_divide(float x, float y)
{
    return x * native_recip(y);
}

ATTR float
native_powr(float x, float y)
{
    return native_exp2(native_log2(x)*y);
}

ATTR float
native_tan(float x)
{
    x *= 0x1.45f306p-3f;
    return native_sin(x) * native_recip(native_cos(x));
}

ATTR float
native_cos(float x)
{
    return __llvm_cos_f32(x);
}

ATTR float
native_exp2(float x)
{
    return __llvm_exp2_f32(x);
}

ATTR float
native_exp(float f) {
    return __llvm_exp2_f32(M_LOG2E_F * f);
}

ATTR float
native_exp10(float f)
{
    return __llvm_exp2_f32(M_LOG2_10_F * f);
}

ATTR float
native_log2(float x) {
    return __llvm_log2_f32(x);
}

ATTR float
native_log(float f)
{
    return __llvm_log2_f32(f) * M_RLOG2_E_F;
}

ATTR float
native_log10(float f)
{
    return __llvm_log2_f32(f) * M_RLOG2_10_F;
}

ATTR float
native_recip(float x) {
    return __llvm_amdgcn_rcp_f32(x);
}

ATTR float
native_rsqrt(float x)
{
    return __llvm_amdgcn_rsq_f32(x);
}

ATTR float
native_sin(float x) {
    return __llvm_sin_f32(x);
}

ATTR float
native_sqrt(float x) {
    return __llvm_sqrt_f32(x);
}

