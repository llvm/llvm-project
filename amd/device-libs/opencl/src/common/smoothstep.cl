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
smoothstep(T edge0, T edge1, T##N x) \
{ \
    T##N t = clamp((x - edge0) / (edge1 - edge0), (T)0, (T)1); \
    return t * t * mad(t, -(T##N)2, (T##N)3); \
} \
 \
ATTR T##N \
smoothstep(T##N edge0, T##N edge1, T##N x) \
{ \
    T##N t = clamp((x - edge0) / (edge1 - edge0), (T)0, (T)1); \
    return t * t * mad(t, -(T##N)2, (T##N)3); \
}

#define GEN1(T) \
ATTR T \
smoothstep(T edge0, T edge1, T x) \
{ \
    T t = clamp((x - edge0) / (edge1 - edge0), (T)0, (T)1); \
    return t * t * mad(t, -(T)2, (T)3); \
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

