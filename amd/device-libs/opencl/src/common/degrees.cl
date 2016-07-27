/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#pragma OPENCL EXTENSION cl_khr_fp16 : enable

#define float_degrees 0x1.ca5dc2p+5f
#define double_degrees 0x1.ca5dc1a63c1f8p+5
#define half_degrees 0x1.ca5dc2p+5h

#define float_radians 0x1.1df46ap-6f
#define double_radians 0x1.1df46a2529d39p-6
#define half_radians 0x1.1df46ap-6h

#define ATTR __attribute__((always_inline, overloadable, const))

#define GENN(N,T,F) \
ATTR T##N \
F(T##N x) \
{ \
    return x * T##_##F; \
}

#define GEN(T,F) \
    GENN(16,T,F) \
    GENN(8,T,F) \
    GENN(4,T,F) \
    GENN(3,T,F) \
    GENN(2,T,F) \
    GENN(,T,F)

GEN(float,radians)
GEN(double,radians)
GEN(half,radians)

GEN(float,degrees)
GEN(double,degrees)
GEN(half,degrees)

