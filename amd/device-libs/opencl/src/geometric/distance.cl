/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#pragma OPENCL EXTENSION cl_khr_fp16 : enable

#define ATTR __attribute__((always_inline, overloadable, const))

#define GENN(N,T) \
ATTR T \
distance(T##N p0, T##N p1) \
{ \
    return length(p0 - p1); \
}

#define GEN(T) \
    GENN(4,T) \
    GENN(3,T) \
    GENN(2,T) \
    GENN(,T)

GEN(float)
GEN(double)
GEN(half)

