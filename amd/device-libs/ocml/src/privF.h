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

#define MATH_SQRT(X) __ocml_sqrt_f32(X)
