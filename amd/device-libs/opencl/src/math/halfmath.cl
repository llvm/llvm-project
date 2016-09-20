/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

// For trigs
extern int __half_red(float, __private float *);
extern float __half_scr(float, __private float *);
extern float __half_tr(float, int);

#define IATTR __attribute__((always_inline, overloadable))
#define CATTR __attribute__((always_inline, overloadable, const))

#if !defined USE_CLP
#define LISTU2(F) F(x.s0), F(x.s1)
#define LISTU3(F) F(x.s0), F(x.s1), F(x.s2)
#define LISTU4(F) LISTU2(F), F(x.s2), F(x.s3)
#define LISTU8(F) LISTU4(F), F(x.s4), F(x.s5), F(x.s6), F(x.s7)
#define LISTU16(F) LISTU8(F), F(x.s8), F(x.s9), F(x.sa), F(x.sb), \
                                F(x.sc), F(x.sd), F(x.se), F(x.sf)

#define EXPUN(N,F) \
IATTR float##N \
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
IATTR float##N \
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

EXPB(half_divide)
EXPB(half_powr)
EXPU(half_cos)
EXPU(half_exp2)
EXPU(half_exp)
EXPU(half_exp10)
EXPU(half_log2)
EXPU(half_log)
EXPU(half_log10)
EXPU(half_recip)
EXPU(half_rsqrt)
EXPU(half_sin)
EXPU(half_sqrt)
EXPU(half_tan)
#endif // !USE_CLP

CATTR float
half_divide(float x, float y)
{
    int c = fabs(y) > 0x1.0p+96f;
    float s = c ? 0x1.0p-32f : 1.0f;
    y *= s;
    return s * native_divide(x, y);
}

IATTR float
half_powr(float x, float y)
{
    return powr(x, y);
}

IATTR float
half_cos(float x)
{
    float dx = fabs(x);
    int ax = as_int(dx);

    float r0;
    int regn = __half_red(dx, &r0);

    float cc;
    float ss = -__half_scr(r0, &cc);

    float c = (regn & 1) != 0 ? ss : cc;
    c = as_float(as_int(c) ^ ((regn > 1) << 31));

    c = ax > 0x47800000 ? 1.0f : c;
    c = ax >= 0x7f800000 ? as_float(0x7fc00000) : c;
    return c;
}

CATTR float
half_exp2(float x)
{
    return native_exp2(x);
}

CATTR float
half_exp(float x)
{
    return native_exp(x);
}

CATTR float
half_exp10(float x)
{
    return native_exp10(x);
}

CATTR float
half_log2(float x)
{
    return native_log2(x);
}

CATTR float
half_log(float x)
{
    return native_log(x);
}

CATTR float
half_log10(float x)
{
    return native_log10(x);
}

CATTR float
half_recip(float x)
{
    return native_recip(x);
}

CATTR float
half_rsqrt(float x)
{
    return native_rsqrt(x);
}

IATTR float
half_sin(float x)
{
    int ix = as_int(x);
    float dx = fabs(x);
    int ax = as_int(dx);

    float r0;
    int regn = __half_red(dx, &r0);

    float cc;
    float ss = __half_scr(r0, &cc);

    float s = (regn & 1) != 0 ? cc : ss;
    s = as_float(as_int(s) ^ ((regn > 1) << 31));

    s = ax > 0x47800000 ? 1.0f : s;
    s = as_float(as_int(s) ^ (ix ^ ax));
    s = x == 0.0f ? x : s;
    s = ax >= 0x7f800000 ? as_float(0x7fc00000) : s;
    return s;
}

CATTR float
half_sqrt(float x)
{
    return native_sqrt(x);
}

IATTR float
half_tan(float x)
{
    int ix = as_int(x);
    float dx = fabs(x);
    int ax = as_int(dx);

    float r0;
    int regn = __half_red(dx, &r0);
    float t = __half_tr(r0, regn);

    t = as_float(as_int(t) ^ (ix ^ ax));
    t = x == 0.0f ? x : t;
    t = ax > 0x47800000 ? 0.0f : t;
    t = ax >= 0x7f800000 ? as_float(0x7fc00000) : t;
    return t;
}

