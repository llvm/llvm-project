/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#pragma OPENCL EXTENSION cl_khr_fp16 : enable

#define ATTR __attribute__((overloadable, const))

#define GENN(N,T) \
ATTR T##N \
mix(T##N x, T##N y, T a) \
{ \
    return mad(y - x, (T##N)a, x); \
} \
 \
ATTR T##N \
mix(T##N x, T##N y, T##N a) \
{ \
    return mad(y - x, a, x); \
}

#define GEN1(T) \
ATTR T \
mix(T x, T y, T a) \
{ \
    return mad(y - x, a, x); \
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

