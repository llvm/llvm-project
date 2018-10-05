/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#define ATTR __attribute__((overloadable, const))

#define GENN(N,T) \
ATTR T##N \
rhadd(T##N x, T##N y) \
{ \
    return convert_##T##N((convert_int##N(x) + convert_int##N(y) + 1) >> 1); \
} \
 \
ATTR u##T##N \
rhadd(u##T##N x, u##T##N y) \
{ \
    return convert_u##T##N((convert_uint##N(x) + convert_uint##N(y) + 1U) >> 1); \
}

#define GEN(T) \
    GENN(16,T) \
    GENN(8,T) \
    GENN(4,T) \
    GENN(3,T) \
    GENN(2,T) \
    GENN(,T)

GEN(char)
GEN(short)

#define LGENN(N,T) \
ATTR T##N \
rhadd(T##N x, T##N y) \
{ \
    T##N c = (x | y) & (T)1; \
    return (x >> 1) + (y >> 1) + c; \
}

#define LGEN(T) \
    LGENN(16,T) \
    LGENN(8,T) \
    LGENN(4,T) \
    LGENN(3,T) \
    LGENN(2,T) \
    LGENN(,T)

LGEN(int)
LGEN(uint)
LGEN(long)
LGEN(ulong)

