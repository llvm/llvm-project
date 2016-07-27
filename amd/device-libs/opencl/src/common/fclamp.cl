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
clamp(T##N x, T lo, T hi) \
{ \
    return fmin(fmax(x, lo), hi); \
} \
 \
ATTR T##N \
clamp(T##N x, T##N lo, T##N hi) \
{ \
    return fmin(fmax(x, lo), hi); \
}

#define GEN1(T) \
ATTR T \
clamp(T x, T lo, T hi) \
{ \
    return fmin(fmax(x, lo), hi); \
}

#define GEN(T) \
    GENN(16,T) \
    GENN(8,T) \
    GENN(4,T) \
    GENN(3,T) \
    GENN(2,T) \
    GEN1(T)

GEN(float)
GEN(double)
GEN(half)

