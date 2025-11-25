/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#define ATTR __attribute__((overloadable, const))

#define GENN(N,T) \
ATTR T##N \
max(T##N x, T y) \
{ \
    T##N vy = (T##N)y; \
    return select(x, vy, x < vy); \
} \
 \
ATTR T##N \
max(T##N x, T##N y) \
{ \
    return select(x, y, x < y); \
}

#define GEN1(T) \
ATTR T \
max(T x, T y) \
{ \
    return x < y ? y : x; \
}

#define GEN(T) \
    GENN(16,T) \
    GENN(8,T) \
    GENN(4,T) \
    GENN(3,T) \
    GENN(2,T) \
    GEN1(T)

GEN(char)
GEN(uchar)
GEN(short)
GEN(ushort)
GEN(int)
GEN(uint)
GEN(long)
GEN(ulong)

