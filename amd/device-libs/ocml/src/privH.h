/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#define MATH_MAD(A,B,C) BUILTIN_FMA_F16(A, B, C)
#define MATH_MAD2(A,B,C) BUILTIN_FMA_2F16(A, B, C)

#define MATH_FAST_RCP(X) BUILTIN_RCP_F16(X)
#define MATH_RCP(X) BUILTIN_DIV_F16(1.0h, X)

#define MATH_FAST_DIV(X, Y) ({ \
    half _fdiv_x = X; \
    half _fdiv_y = Y; \
    half _fdiv_ret = _fdiv_x * BUILTIN_RCP_F16(_fdiv_y); \
    _fdiv_ret; \
})
#define MATH_DIV(X,Y) BUILTIN_DIV_F16(X, Y)

#define MATH_FAST_SQRT(X) BUILTIN_SQRT_F16(X)
#define MATH_SQRT(X) ((half)BUILTIN_SQRT_F32((float)(X)))

