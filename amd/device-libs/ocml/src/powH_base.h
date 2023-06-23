/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

static bool
samesign(half x, half y)
{
    return ((AS_USHORT(x) ^ AS_USHORT(y)) & (ushort)0x8000) == (ushort)0;
}

REQUIRES_16BIT_INSTS CONSTATTR half
#if defined(COMPILING_POWR)
MATH_MANGLE(powr)(half x, half y)
#elif defined(COMPILING_POWN)
MATH_MANGLE(pown)(half x, int ny)
#elif defined(COMPILING_ROOTN)
MATH_MANGLE(rootn)(half x, int ny)
#else
MATH_MANGLE(pow)(half x, half y)
#endif
{
    half ax = BUILTIN_ABS_F16(x);

#if defined(COMPILING_POWN)
    float fy = (float)ny;
#elif defined(COMPILING_ROOTN)
    float fy = BUILTIN_AMDGPU_RCP_F32((float)ny);
#else
    float fy = (float)y;
#endif

    float p = BUILTIN_AMDGPU_EXP2_F32(fy * BUILTIN_AMDGPU_LOG2_F32((float)ax));

    // Classify y:
    //   inty = 0 means not an integer.
    //   inty = 1 means odd integer.
    //   inty = 2 means even integer.

#if defined(COMPILING_POWN) || defined(COMPILING_ROOTN)
    int inty = 2 - (ny & 1);
#else
    half ay = BUILTIN_ABS_F16(y);
    int inty;
    {
        half tay = BUILTIN_TRUNC_F16(ay);
        inty = ay == tay;
        inty += inty & (BUILTIN_FRACTION_F16(tay*0.5h) == 0.0h);
    }
#endif

    half ret = BUILTIN_COPYSIGN_F16((half)p, ((inty == 1) & (x < 0.0h)) ? -0.0f : 0.0f);

    // Now all the edge cases
#if defined COMPILING_POWR
    half iz = y < 0.0h ? PINF_F16 : 0.0h;
    half zi = y < 0.0h ? 0.0h : PINF_F16;

    if (x == 0.0h)
        ret = iz;

    if (BUILTIN_ISINF_F16(x))
        ret = zi;

    if (BUILTIN_ISINF_F16(y))
        ret = ax < 1.0h ? iz : zi;

    if (y == 0.0h)
        ret = x == 0.0h || BUILTIN_ISINF_F16(x) ? QNAN_F16 : 1.0h;

    if (x == 1.0h)
        ret = BUILTIN_ISINF_F16(y) ? QNAN_F16 : 1.0h;

    if (x < 0.0h || BUILTIN_ISUNORDERED_F16(x, y))
        ret = QNAN_F16;
#elif defined COMPILING_POWN
    if (BUILTIN_ISINF_F16(ax) || x == 0.0h)
        ret = BUILTIN_COPYSIGN_F16((x == 0.0h) ^ (ny < 0) ? 0.0h : PINF_F16,
                                   inty == 1 ? x : 0.0h);

    if (BUILTIN_ISNAN_F16(x))
        ret = QNAN_F16;

    if (ny == 0)
        ret = 1.0h;
#elif defined COMPILING_ROOTN
    if (BUILTIN_ISINF_F16(ax) || x == 0.0h)
        ret = BUILTIN_COPYSIGN_F16((x == 0.0h) ^ (ny < 0) ? 0.0h : PINF_F16,
                                   inty == 1 ? x : 0.0h);

    if ((x < 0.0h && inty != 1) || ny == 0)
        ret = QNAN_F16;
#else
    if (x < 0.0h && !inty)
        ret = QNAN_F16;

    if (BUILTIN_ISINF_F16(ay))
        ret = ax == 1.0h ? ax : (samesign(y, ax - 1.0h) ? ay : 0.0h);

    if (BUILTIN_ISINF_F16(ax) || x == 0.0h)
        ret = BUILTIN_COPYSIGN_F16((x == 0.0h) ^ (y < 0.0h) ? 0.0h : PINF_F16,
                                   inty == 1 ? x : 0.0h);

    if (BUILTIN_ISUNORDERED_F16(x, y))
        ret = QNAN_F16;

    if (x == 1.0h || y == 0.0h)
        ret = 1.0h;
#endif

    return ret;
}

