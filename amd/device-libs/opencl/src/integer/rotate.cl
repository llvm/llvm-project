/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "int.h"

#define ATTR __attribute__((overloadable, const))

#define char_bits 8
#define short_bits 16
#define int_bits 32
#define long_bits 64

#define GENN(N,T) \
ATTR T##N \
rotate(T##N x, T##N y) \
{ \
    uint##N s = convert_uint##N(as_u##T##N(y)) & (uint)(T##_bits - 1); \
    uint##N v = convert_uint##N(as_u##T##N(x)); \
    return convert_##T##N((v << s) | (v >> (T##_bits - s))); \
} \
 \
ATTR u##T##N \
rotate(u##T##N x, u##T##N y) \
{ \
    uint##N s = convert_uint##N(y) & (uint)(T##_bits - 1); \
    uint##N v = convert_uint##N(x); \
    return convert_u##T##N((v << s) | (v >> ((uint)T##_bits - s))); \
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
rotate(T##N x, T##N y) \
{ \
    u##T##N s = as_u##T##N(y) & (u##T)(T##_bits - 1); \
    u##T##N v = as_u##T##N(x); \
    return as_##T##N((v << s) | (v >> ((u##T)T##_bits - s))); \
} \
 \
ATTR u##T##N \
rotate(u##T##N x, u##T##N y) \
{ \
    y &= (u##T)(T##_bits - 1); \
    return (x << y) | (x >> ((u##T)T##_bits - y)); \
}

#define LGEN(T) \
    LGENN(16,T) \
    LGENN(8,T) \
    LGENN(4,T) \
    LGENN(3,T) \
    LGENN(2,T) \
    LGENN(,T)

LGEN(int)
LGEN(long)

