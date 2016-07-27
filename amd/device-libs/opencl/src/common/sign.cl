/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#pragma OPENCL EXTENSION cl_khr_fp16 : enable

#define ATTR __attribute__((always_inline, overloadable, const))

#define GENN(N,T) \
ATTR T##N \
sign(T##N x) \
{ \
    return copysign(isnan(x) | (x == (T##N)0) ? (T##N)0 : (T##N)1, x); \
}

#define GEN(T) \
    GENN(16,T) \
    GENN(8,T) \
    GENN(4,T) \
    GENN(3,T) \
    GENN(2,T) \
    GENN(,T)

GEN(float)
GEN(double)
GEN(half)

