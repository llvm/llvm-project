/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

static float compute_expylnx_f16(half ax, half y)
{
    return BUILTIN_AMDGPU_EXP2_F32((float)y * BUILTIN_AMDGPU_LOG2_F32((float)ax));
}

static bool is_integer(half ay)
{
    return BUILTIN_TRUNC_F16(ay) == ay;
}

static bool is_even_integer(half ay) {
    // Even integers are still integers after division by 2.
    return is_integer(0.5h * ay);
}

static bool is_odd_integer(half ay) {
    return is_integer(ay) && !is_even_integer(ay);
}

#if defined(COMPILING_POW)

CONSTATTR half
MATH_MANGLE(pow)(half x, half y)
{
    if (x == 1.0h)
        y = 1.0h;
    if (y == 0.0h)
        x = 1.0h;

    half ax = BUILTIN_ABS_F16(x);
    float p = compute_expylnx_f16(ax, y);

    bool is_odd_y = is_odd_integer(y);
    half ret = BUILTIN_COPYSIGN_F16((half)p, is_odd_y ? x : 1.0f);

    // Now all the edge cases
    if (x < 0.0h && !is_integer(y))
        ret = QNAN_F16;

    half ay = BUILTIN_ABS_F16(y);
    if (BUILTIN_ISINF_F16(ay)) {
        // FIXME: Missing backend optimization to save on
        // materialization cost of mixed sign constant infinities.
        bool y_is_neg_inf = y != ay;
        ret = ax == 1.0h ? ax : ((ax < 1.0h) ^ y_is_neg_inf ? 0.0h : ay);
    }

    if (BUILTIN_ISINF_F16(ax) || x == 0.0h) {
        ret = BUILTIN_COPYSIGN_F16((x == 0.0h) ^ (y < 0.0h) ? 0.0h : PINF_F16,
                                   is_odd_y ? x : 0.0h);
    }

    if (BUILTIN_ISUNORDERED_F16(x, y))
        ret = QNAN_F16;

    return ret;
}

#elif defined(COMPILING_POWR)

CONSTATTR half
MATH_MANGLE(powr)(half x, half y)
{
    if (x < 0.0h)
        x = QNAN_F16;

    half ret = (half)compute_expylnx_f16(x, y);

    // Now all the edge cases
    half iz = y < 0.0h ? PINF_F16 : 0.0h;
    half zi = y < 0.0h ? 0.0h : PINF_F16;

    if (x == 0.0h)
        ret = y == 0.0h ? QNAN_F16 : iz;

    if (x == PINF_F16 && y != 0.0h)
        ret = zi;

    if (BUILTIN_ISINF_F16(y) && x != 1.0h)
        ret = x < 1.0h ? iz : zi;

    if (BUILTIN_ISUNORDERED_F16(x, y))
        ret = QNAN_F16;

    return ret;
}


#elif defined(COMPILING_POWN)

CONSTATTR half
MATH_MANGLE(pown)(half x, int ny)
{
    if (ny == 0)
        x = 1.0h;

    half ax = BUILTIN_ABS_F16(x);

    float fy = (float)ny;

    float p = BUILTIN_AMDGPU_EXP2_F32(fy * BUILTIN_AMDGPU_LOG2_F32((float)ax));

    bool is_odd_y = ny & 1;

    half ret = BUILTIN_COPYSIGN_F16((half)p, is_odd_y ? x : 1.0f);

    // Now all the edge cases
    if (BUILTIN_ISINF_F16(ax) || x == 0.0h)
        ret = BUILTIN_COPYSIGN_F16((x == 0.0h) ^ (ny < 0) ? 0.0h : PINF_F16,
                                   is_odd_y ? x : 0.0h);

    return ret;
}

#elif defined(COMPILING_ROOTN)

CONSTATTR half
MATH_MANGLE(rootn)(half x, int ny)
{
    half ax = BUILTIN_ABS_F16(x);

    float fy = BUILTIN_AMDGPU_RCP_F32((float)ny);

    float p = BUILTIN_AMDGPU_EXP2_F32(fy * BUILTIN_AMDGPU_LOG2_F32((float)ax));

    bool is_odd_y = ny & 1;

    half ret = BUILTIN_COPYSIGN_F16((half)p, is_odd_y ? x : 1.0f);

    // Now all the edge cases
    if (BUILTIN_ISINF_F16(ax) || x == 0.0h)
        ret = BUILTIN_COPYSIGN_F16((x == 0.0h) ^ (ny < 0) ? 0.0h : PINF_F16,
                                   is_odd_y ? x : 0.0h);

    if ((x < 0.0h && !is_odd_y) || ny == 0)
        ret = QNAN_F16;

    return ret;
}

#else
#error missing function macro
#endif
