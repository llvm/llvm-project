/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#pragma OPENCL EXTENSION cl_khr_fp16 : enable

#define ATTR __attribute__((always_inline, overloadable))

#define GENN(N,T) \
ATTR void \
prefetch(const __global T##N *p, size_t n) \
{ \
}

#define GEN(T) \
    GENN(16,T) \
    GENN(8,T) \
    GENN(4,T) \
    GENN(3,T) \
    GENN(2,T) \
    GENN(,T)

GEN(char)
GEN(uchar)
GEN(short)
GEN(ushort)
GEN(int)
GEN(uint)
GEN(long)
GEN(ulong)
GEN(float)
GEN(double)
GEN(half)

