/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#define MATH_MAD(A,B,C) BUILTIN_MAD_F32(A, B, C)
#define MATH_MAD2(A,B,C) BUILTIN_MAD_2F32(A, B, C)

#define MATH_FAST_RCP(X) BUILTIN_AMDGPU_RCP_F32(X)
#define MATH_RCP(X) BUILTIN_DIV_F32(1.0f, X)

#define MATH_FAST_DIV(X, Y) ({ \
    float _fdiv_x = X; \
    float _fdiv_y = Y; \
    float _fdiv_ret = _fdiv_x * BUILTIN_AMDGPU_RCP_F32(_fdiv_y); \
    _fdiv_ret; \
})
#define MATH_DIV(X,Y) BUILTIN_DIV_F32(X, Y)

#define MATH_FAST_SQRT(X) BUILTIN_AMDGPU_SQRT_F32(X)

#define MATH_SQRT(X) ({ \
    float _sqrt_x = X; \
    bool _sqrt_b = _sqrt_x < 0x1.0p-96f; \
    _sqrt_x *= _sqrt_b ? 0x1.0p+32f : 1.0f; \
    float _sqrt_s; \
    if (!DAZ_OPT()) { \
        _sqrt_s = BUILTIN_AMDGPU_SQRT_F32(_sqrt_x); \
        float _sqrt_sp = AS_FLOAT(AS_INT(_sqrt_s) - 1); \
        float _sqrt_ss = AS_FLOAT(AS_INT(_sqrt_s) + 1); \
        float _sqrt_vp = BUILTIN_FMA_F32(-_sqrt_sp, _sqrt_s, _sqrt_x); \
        float _sqrt_vs = BUILTIN_FMA_F32(-_sqrt_ss, _sqrt_s, _sqrt_x); \
        _sqrt_s = _sqrt_vp <= 0.0f ? _sqrt_sp : _sqrt_s; \
        _sqrt_s = _sqrt_vs >  0.0f ? _sqrt_ss : _sqrt_s; \
    } else { \
        float _sqrt_r = BUILTIN_AMDGPU_RSQRT_F32(_sqrt_x); \
        _sqrt_s = _sqrt_x * _sqrt_r; \
        float _sqrt_h = 0.5f * _sqrt_r; \
        float _sqrt_e = BUILTIN_FMA_F32(-_sqrt_h, _sqrt_s, 0.5f); \
        _sqrt_h = BUILTIN_FMA_F32(_sqrt_h, _sqrt_e, _sqrt_h); \
        _sqrt_s = BUILTIN_FMA_F32(_sqrt_s, _sqrt_e, _sqrt_s); \
        float _sqrt_d = BUILTIN_FMA_F32(-_sqrt_s, _sqrt_s, _sqrt_x); \
        _sqrt_s = BUILTIN_FMA_F32(_sqrt_d, _sqrt_h, _sqrt_s); \
    } \
    _sqrt_s *= _sqrt_b ? 0x1.0p-16f : 1.0f; \
    _sqrt_s = BUILTIN_CLASS_F32(_sqrt_x, CLASS_PZER|CLASS_NZER|CLASS_PINF) ? _sqrt_x : _sqrt_s; \
    _sqrt_s; \
})
