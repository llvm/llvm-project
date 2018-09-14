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
step(T edge, T##N x) \
{ \
    return select((T##N)1, (T##N)0, x < edge); \
} \
 \
ATTR T##N \
step(T##N edge, T##N x) \
{ \
    return select((T##N)1, (T##N)0, x < edge); \
}

#define GEN1(T) \
ATTR T \
step(T edge, T x) \
{ \
    return x < edge ? (T)0 : (T)1; \
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

