/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#define ATTR __attribute__((always_inline, const))


// TODO - remove these when these conversions are ordinary LLVM conversions

ATTR uint
__cvt_f16_rtn_f32(float a)
{
    uint u = as_uint(a);
    uint um = u & 0x7fffffU;
    int e = (int)((u >> 23) & 0xff) - 127 + 15;
    int ds = max(0, min(19, 1 - e));
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
    return s | v;
}

ATTR uint
__cvt_f16_rtp_f32(float a)
{
    uint u = as_uint(a);
    uint um = u & 0x7fffffU;
    int e = (int)((u >> 23) & 0xff) - 127 + 15;
    int ds = max(0, min(19, 1 - e));
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
    return s | v;
}

ATTR uint
__cvt_f16_rtz_f32(float a)
{
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
    return s | v;
}

ATTR uint
__cvt_f16_rte_f64(double a)
{
    ulong u = as_ulong(a);
    uint uh = u >> 32;
    int e = (int)((uh >> 20) & 0x7ff) - 1023 + 15;
    uint m = ((uh >> 8) & 0xffe) | (((uh & 0x1ff) | (uint)u) != 0);
    uint i = 0x7c00 | (m != 0 ? 0x0200 : 0);
    uint n = ((uint)e << 12) | m;
    uint s = (uh >> 16) & 0x8000;
    int b = clamp(1-e, 0, 13);
    uint d = (0x1000 | m) >> b;
    d |= (d << b) != (0x1000 | m);
    uint v = e < 1 ? d : n;
    v = (v >> 2) + ((v & 0x7) == 3 | (v & 0x7) > 5);
    v = e > 30 ? 0x7c00 : v;
    v = e == 1039 ? i : v;
    return s | v;
}

ATTR uint
__cvt_f16_rtn_f64(double a)
{
    ulong u = as_ulong(a);
    uint uh = u >> 32;
    int e = (int)((uh >> 20) & 0x7ff) - 1023 + 15;
    uint m = ((uh >> 9) & 0x7fe) | (((uh & 0x3ff) | (uint)u) != 0);
    uint i = 0x7c00 | (m != 0 ? 0x0200 : 0);
    uint n = ((uint)e << 11) | m;
    uint s = (uh >> 16) & 0x8000;
    uint vp = 0x7bff + (s >> 15);
    int b = clamp(1-e, 0, 12);
    uint d = (0x800 | m) >> b;
    d |= (d << b) != (0x800 | m);
    uint v = e < 1 ? d : n;
    v = (v >> 1) + (v & 1 & (s >> 15));
    v = e > 30 ? vp : v;
    v = e == 1039 ? i : v;
    v = e == -1008 & m == 0 ? 0 : v;
    return s | v;
}

ATTR uint
__cvt_f16_rtp_f64(double a)
{
    ulong u = as_ulong(a);
    uint uh = u >> 32;
    int e = (int)((uh >> 20) & 0x7ff) - 1023 + 15;
    uint m = ((uh >> 9) & 0x7fe) | (((uh & 0x3ff) | (uint)u) != 0);
    uint i = 0x7c00 | (m != 0 ? 0x0200 : 0);
    uint n = ((uint)e << 11) | m;
    uint s = (uh >> 16) & 0x8000;
    uint vp = 0x7c00 - (s >> 15);
    int b = clamp(1-e, 0, 12);
    uint d = (0x800 | m) >> b;
    d |= (d << b) != (0x800 | m);
    uint v = e < 1 ? d : n;
    v = (v >> 1) + (v & 1 & ((s >> 15) ^ 1));
    v = e > 30 ? vp : v;
    v = e == 1039 ? i : v;
    v = e == -1008 & m == 0 ? 0 : v;
    return s | v;
}

ATTR uint
__cvt_f16_rtz_f64(double a)
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
    return ((uh >> 16) & 0x8000) | v;
}

