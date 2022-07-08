/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "ocml.h"
#include "builtins.h"
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#define ATTR __attribute__((const))
#define AATTR(S) __attribute__((const, alias(S)))
#undef AVOID_FP

ATTR half
OCML_MANGLE_F32(cvtrtn_f16)(float a)
{
#if defined AVOID_FP
    uint u = as_uint(a);
    uint um = u & 0x7fffffU;
    int e = (int)((u >> 23) & 0xff) - 127 + 15;
    int ds = BUILTIN_CLAMP_S32(1-e, 0, 19);
    uint t = (um | (e > -112 ? 0x800000 : 0)) << (19 - ds);
    uint s = (u >> 16) & 0x8000;
    uint m = (u >> 13) & 0x3ff;
    uint i = 0x7c00 | m | (um ? 0x0200 : 0);
    uint n = ((uint)e << 10) | m;
    uint d = (0x400 | m) >> ds;
    uint v = e < 1 ? d : n;
    v += (s >> 15) & (t > 0U);
    uint j = 0x7bff + (s >> 15);
    v = e > 30 ? j : v;
    v = e == 143 ? i : v;
    return AS_HALF((ushort)(s | v));
#else
    half r = (half)a;
    half p = OCML_MANGLE_F16(pred)(r);
    return (float)r > a ? p : r;
#endif
}

ATTR half
OCML_MANGLE_F32(cvtrtp_f16)(float a)
{
#if defined AVOID_FP
    uint u = as_uint(a);
    uint um = u & 0x7fffffU;
    int e = (int)((u >> 23) & 0xff) - 127 + 15;
    int ds = BUILTIN_CLAMP_S32(1-e, 0, 19);
    uint t = (um | (e > -112 ? 0x800000 : 0)) << (19 - ds);
    uint s = (u >> 16) & 0x8000;
    uint m = (u >> 13) & 0x3ff;
    uint i = 0x7c00 | m | (um ? 0x0200 : 0);
    uint n = ((uint)e << 10) | m;
    uint d = (0x400 | m) >> ds;
    uint v = e < 1 ? d : n;
    v += ~(s >> 15) & (t > 0U);
    uint j = 0x7c00 - (s >> 15);
    v = e > 30 ? j : v;
    v = e == 143 ? i : v;
    return AS_HALF((ushort)(s | v));
#else
    half r = (half)a;
    half s = OCML_MANGLE_F16(succ)(r);
    return (float)r < a ? s : r;
#endif
}

ATTR half
OCML_MANGLE_F32(cvtrtz_f16)(float a)
{
#if defined AVOID_FP
    uint u = as_uint(a);
    uint um = u & 0x7fffffU;
    int e = (int)((u >> 23) & 0xff) - 127 + 15;
    uint s = (u >> 16) & 0x8000;
    uint m = (u >> 13) & 0x3ff;
    uint i = 0x7c00 | m | (um ? 0x0200 : 0);
    uint n = ((uint)e << 10) | m;
    uint d = (0x400 | m) >> (1 - e);
    uint v = e > 30 ? 0x7bff : n;
    v = e == 143 ? i : v;
    v = e < 1 ? d : v;
    v = e < -10 ? 0 : v;
    return AS_HALF((ushort)(s | v));
#else
    float aa = BUILTIN_ABS_F32(a);
    half r = (half)a;
    half ar = BUILTIN_ABS_F16(r);
    half z = OCML_MANGLE_F16(nextafter)(r, 0.0h);
    return aa < (float)ar ? z : r;
#endif
}

ATTR half
OCML_MANGLE_F64(cvtrte_f16)(double a)
{
    ulong u = as_ulong(a);
    uint uh = u >> 32;
    int e = (int)((uh >> 20) & 0x7ff) - 1023 + 15;
    uint m = ((uh >> 8) & 0xffe) | (((uh & 0x1ff) | (uint)u) != 0);
    uint i = 0x7c00 | (m != 0 ? 0x0200 : 0);
    uint n = ((uint)e << 12) | m;
    uint s = (uh >> 16) & 0x8000;
    int b = BUILTIN_CLAMP_S32(1-e, 0, 13);
    uint d = (0x1000 | m) >> b;
    d |= (d << b) != (0x1000 | m);
    uint v = e < 1 ? d : n;
    v = (v >> 2) + ((v & 0x7) == 3 | (v & 0x7) > 5);
    v = e > 30 ? 0x7c00 : v;
    v = e == 1039 ? i : v;
    return AS_HALF((ushort)(s | v));
}

ATTR half
OCML_MANGLE_F64(cvtrtn_f16)(double a)
{
    ulong u = as_ulong(a);
    uint uh = u >> 32;
    int e = (int)((uh >> 20) & 0x7ff) - 1023 + 15;
    uint m = ((uh >> 9) & 0x7fe) | (((uh & 0x3ff) | (uint)u) != 0);
    uint i = 0x7c00 | (m != 0 ? 0x0200 : 0);
    uint n = ((uint)e << 11) | m;
    uint s = (uh >> 16) & 0x8000;
    uint vp = 0x7bff + (s >> 15);
    int b = BUILTIN_CLAMP_S32(1-e, 0, 12);
    uint d = (0x800 | m) >> b;
    d |= (d << b) != (0x800 | m);
    uint v = e < 1 ? d : n;
    v = (v >> 1) + (v & 1 & (s >> 15));
    v = e > 30 ? vp : v;
    v = e == 1039 ? i : v;
    v = (e == -1008 & m == 0) ? 0 : v;
    return AS_HALF((ushort)(s | v));
}

ATTR half
OCML_MANGLE_F64(cvtrtp_f16)(double a)
{
    ulong u = as_ulong(a);
    uint uh = u >> 32;
    int e = (int)((uh >> 20) & 0x7ff) - 1023 + 15;
    uint m = ((uh >> 9) & 0x7fe) | (((uh & 0x3ff) | (uint)u) != 0);
    uint i = 0x7c00 | (m != 0 ? 0x0200 : 0);
    uint n = ((uint)e << 11) | m;
    uint s = (uh >> 16) & 0x8000;
    uint vp = 0x7c00 - (s >> 15);
    int b = BUILTIN_CLAMP_S32(1-e, 0, 12);
    uint d = (0x800 | m) >> b;
    d |= (d << b) != (0x800 | m);
    uint v = e < 1 ? d : n;
    v = (v >> 1) + (v & 1 & ((s >> 15) ^ 1));
    v = e > 30 ? vp : v;
    v = e == 1039 ? i : v;
    v = (e == -1008 & m == 0) ? 0 : v;
    return AS_HALF((ushort)(s | v));
}

ATTR half
OCML_MANGLE_F64(cvtrtz_f16)(double a)
{
    ulong u = as_ulong(a);
    uint uh = u >> 32;
    uint m = ((uh >> 9) & 0x7fe) | (((uh & 0x3ff) | (uint)u) != 0);
    int e = (int)((uh >> 20) & 0x7ff) - 1023 + 15;
    uint i = 0x7c00 | (m != 0 ? 0x0200 : 0);
    m >>= 1;
    uint d = (0x400 | m) >> (1 - e);
    uint n = ((uint)e << 10) | m;
    uint v = e > 30 ? 0x7bff : n;
    v = e == 1039 ? i : v;
    v = e < 1 ? d : v;
    v = e < -10 ? 0 : v;
    return AS_HALF((ushort)(((uh >> 16) & 0x8000) | v));
}

ATTR float
OCML_MANGLE_F64(cvtrtn_f32)(double a)
{
#if defined AVOID_FP
    ulong u = as_ulong(a);
    ulong um = u & 0xfffffffffffffUL;
    int e = (int)((u >> 52) & 0x7ff) - 1023 + 127;
    int ds = BUILTIN_CLAMP_S32(1-e, 0, 31);
    ulong t = (um | (e > -896 ? 0x0010000000000000UL : 0UL)) << (35 - ds);
    uint s = (uint)(u >> 32) & 0x80000000;
    uint m = (uint)(u >> 29) & 0x7fffff;
    uint i = 0x7f800000 | m | (um ? 0x00400000 : 0U);
    uint n = ((uint)(e << 23)) | m;
    uint d = (0x800000 | m) >> ds;
    uint v = e < 1 ? d : n;
    v += (s >> 31) & (t > 0UL);
    uint j = 0x7f7fffff + (s >> 31);
    v = e > 254 ? j : v;
    v = e == 1151 ? i : v;
    return as_float(s | v);
#else
    float r = (float)a;
    float p = OCML_MANGLE_F32(pred)(r);
    return (double)r > a ? p : r;
#endif
}

ATTR float
OCML_MANGLE_F64(cvtrtp_f32)(double a)
{
#if defined AVOID_FP
    ulong u = as_ulong(a);
    ulong um = u & 0xfffffffffffffUL;
    int e = (int)((u >> 52) & 0x7ff) - 1023 + 127;
    int ds = BUILTIN_CLAMP_S32(1-e, 0, 31);
    ulong t = (um | (e > -896 ? 0x0010000000000000UL : 0UL)) << (35 - ds);
    uint s = (uint)(u >> 32) & 0x80000000;
    uint m = (uint)(u >> 29) & 0x7fffff;
    uint i = 0x7f800000 | m | (um ? 0x00400000 : 0U);
    uint n = ((uint)(e << 23)) | m;
    uint d = (0x800000 | m) >> ds;
    uint v = e < 1 ? d : n;
    v += ~(s >> 31) & (t > 0UL);
    uint j = 0x7f800000 - (s >> 31);
    v = e > 254 ? j : v;
    v = e == 1151 ? i : v;
    return as_float(s | v);
#else
    float r = (float)a;
    float s = OCML_MANGLE_F32(succ)(r);
    return (double)r < a ? s : r;
#endif
}

ATTR float
OCML_MANGLE_F64(cvtrtz_f32)(double a)
{
#if defined AVOID_FP
    ulong u = as_ulong(a);
    ulong um = u & 0xfffffffffffffUL;
    int e = (int)((u >> 52) & 0x7ff) - 1023 + 127;
    uint s = (uint)(u >> 32) & 0x80000000;
    uint m = (uint)(u >> 29) & 0x7fffff;
    uint i = 0x7f800000 | m | (um ? 0x00400000 : 0U);
    uint n = ((uint)(e << 23)) | m;
    uint d = (0x800000 | m) >> (1 - e);
    uint v = e > 254 ? 0x7f7fffff : n;
    v = e == 1151 ? i : v;
    v = e < 1 ? d : v;
    v = e < -23 ? 0 : v;
    return as_float(s | v);
#else
    double aa = BUILTIN_ABS_F64(a);
    float r = (float)a;
    float ar = BUILTIN_ABS_F32(r);
    float z = OCML_MANGLE_F32(nextafter)(r, 0.0f);
    return aa < (double)ar ? z : r;
#endif
}

ATTR float
OCML_MANGLE_S32(cvtrtn_f32)(int i)
{
    int s = i >> 31;
    uint u = AS_UINT((i + s) ^ s);
    uint lz = BUILTIN_CLZ_U32(u);
    uint e = 127U + 31U - lz;
    e = u ? e : 0;
    u = (u << lz) & 0x7fffffffU;
    uint t = u & 0xffU;
    u = (e << 23) | (u >> 8);
    return AS_FLOAT((u + ((s & t) > 0)) | (s & 0x80000000));
}

ATTR float
OCML_MANGLE_S32(cvtrtp_f32)(int i)
{
    int s = i >> 31;
    uint u = AS_UINT((i + s) ^ s);
    uint lz = BUILTIN_CLZ_U32(u);
    uint e = 127U + 31U - lz;
    e = u ? e : 0;
    u = (u << lz) & 0x7fffffffU;
    uint t = u & 0xffU;
    u = (e << 23) | (u >> 8);
    return AS_FLOAT((u + ((~s & t) > 0)) | (s & 0x80000000));
}

ATTR float
OCML_MANGLE_S32(cvtrtz_f32)(int i)
{
    int s = i >> 31;
    uint u = AS_UINT((i + s) ^ s);
    uint lz = BUILTIN_CLZ_U32(u);
    uint e = 127U + 31U - lz;
    e = u ? e : 0;
    u = (u << lz) & 0x7fffffffU;
    u = (e << 23) | (u >> 8);
    return AS_FLOAT(u | (s & 0x80000000));
}

ATTR static float
cvt1f4_zu4(uint u)
{
    uint lz = BUILTIN_CLZ_U32(u);
    uint e = 127U + 31U - lz;
    e = u ? e : 0;
    u = (u << lz) & 0x7fffffffU;
    return AS_FLOAT((e << 23) | (u >> 8));
}
extern AATTR("cvt1f4_zu4") float OCML_MANGLE_U32(cvtrtn_f32)(uint);
extern AATTR("cvt1f4_zu4") float OCML_MANGLE_U32(cvtrtz_f32)(uint);

ATTR float
OCML_MANGLE_U32(cvtrtp_f32)(uint u)
{
    uint lz = BUILTIN_CLZ_U32(u);
    uint e = 127U + 31U - lz;
    e = u ? e : 0;
    u = (u << lz) & 0x7fffffffU;
    uint t = u & 0xffU;
    u = (e << 23) | (u >> 8);
    return AS_FLOAT(u + (t > 0));
}

ATTR float
OCML_MANGLE_S64(cvtrtn_f32)(long l)
{
    long s = l >> 63;
    ulong u = AS_ULONG((l + s) ^ s);
    uint lz = BUILTIN_CLZ_U64(u);
    uint e = 127U + 63U - lz;
    e = u ? e : 0;
    u = (u << lz) & 0x7fffffffffffffffUL;
    ulong t = u & 0xffffffffffUL;
    uint v = (e << 23) | (uint)(u >> 40);
    return AS_FLOAT((v + ((s & t) > 0)) | ((uint)s & 0x80000000));
}

ATTR float
OCML_MANGLE_S64(cvtrtp_f32)(long l)
{
    long s = l >> 63;
    ulong u = AS_ULONG((l + s) ^ s);
    uint lz = BUILTIN_CLZ_U64(u);
    uint e = 127U + 63U - lz;
    e = u ? e : 0;
    u = (u << lz) & 0x7fffffffffffffffUL;
    ulong t = u & 0xffffffffffUL;
    uint v = (e << 23) | (uint)(u >> 40);
    return AS_FLOAT((v + ((~s & t) > 0)) | ((uint)s & 0x80000000));
}

ATTR float
OCML_MANGLE_S64(cvtrtz_f32)(long l)
{
    long s = l >> 63;
    ulong u = AS_ULONG((l + s) ^ s);
    uint lz = BUILTIN_CLZ_U64(u);
    uint e = 127U + 63U - lz;
    e = u ? e : 0;
    u = (u << lz) & 0x7fffffffffffffffUL;
    uint v = (e << 23) | (uint)(u >> 40);
    return AS_FLOAT(v | ((uint)s & 0x80000000));
}

ATTR static float
cvt1f4_zu8(ulong u)
{
    uint lz = BUILTIN_CLZ_U64(u);
    uint e = 127U + 63U - lz;
    e = u ? e : 0;
    u = (u << lz) & 0x7fffffffffffffffUL;
    return AS_FLOAT((e << 23) | (uint)(u >> 40));
}
extern AATTR("cvt1f4_zu8") float OCML_MANGLE_U64(cvtrtn_f32)(ulong);
extern AATTR("cvt1f4_zu8") float OCML_MANGLE_U64(cvtrtz_f32)(ulong);

ATTR float
OCML_MANGLE_U64(cvtrtp_f32)(ulong u)
{
    uint lz = BUILTIN_CLZ_U64(u);
    uint e = 127U + 63U - lz;
    e = u ? e : 0;
    u = (u << lz) & 0x7fffffffffffffffUL;
    ulong t = u & 0xffffffffffUL;
    uint v = (e << 23) | (uint)(u >> 40);
    return AS_FLOAT(v + (t > 0));
}

ATTR double
OCML_MANGLE_S64(cvtrtn_f64)(long l)
{
    long s = l >> 63;
    ulong u = AS_ULONG((l + s) ^ s);
    uint lz = BUILTIN_CLZ_U64(u);
    uint e = 1023U + 63U - lz;
    e = u ? e : 0;
    u = (u << lz) & 0x7fffffffffffffffUL;
    ulong t = u & 0x7ffUL;
    u = ((ulong)e << 52) | (u >> 11);
    return AS_DOUBLE((u + ((s & t) > 0)) | ((ulong)s & 0x8000000000000000UL));
}

ATTR double
OCML_MANGLE_S64(cvtrtp_f64)(long l)
{
    long s = l >> 63;
    ulong u = AS_ULONG((l + s) ^ s);
    uint lz = BUILTIN_CLZ_U64(u);
    uint e = 1023U + 63U - lz;
    e = u ? e : 0;
    u = (u << lz) & 0x7fffffffffffffffUL;
    ulong t = u & 0x7ffUL;
    u = ((ulong)e << 52) | (u >> 11);
    return AS_DOUBLE((u + ((~s & t) > 0)) | ((ulong)s & 0x8000000000000000UL));
}

ATTR double
OCML_MANGLE_S64(cvtrtz_f64)(long l)
{
    long s = l >> 63;
    ulong u = AS_ULONG((l + s) ^ s);
    uint lz = BUILTIN_CLZ_U64(u);
    uint e = 1023U + 63U - lz;
    e = u ? e : 0;
    u = (u << lz) & 0x7fffffffffffffffUL;
    u = ((ulong)e << 52) | (u >> 11);
    return AS_DOUBLE(u | ((ulong)s & 0x8000000000000000UL));
}

ATTR static double
cvt1f8_zu8(ulong u)
{
    uint lz = BUILTIN_CLZ_U64(u);
    uint e = 1023U + 63U - lz;
    e = u ? e : 0;
    u = (u << lz) & 0x7fffffffffffffffUL;
    return AS_DOUBLE(((ulong)e << 52) | (u >> 11));
}
AATTR("cvt1f8_zu8") double OCML_MANGLE_U64(cvtrtn_f64)(ulong);
AATTR("cvt1f8_zu8") double OCML_MANGLE_U64(cvtrtz_f64)(ulong);

ATTR double
OCML_MANGLE_U64(cvtrtp_f64)(ulong u)
{
    uint lz = BUILTIN_CLZ_U64(u);
    uint e = 1023U + 63U - lz;
    e = u ? e : 0;
    u = (u << lz) & 0x7fffffffffffffffUL;
    ulong t = u & 0x7ffUL;
    u = ((ulong)e << 52) | (u >> 11);
    return AS_DOUBLE(u + (t > 0UL));
}

