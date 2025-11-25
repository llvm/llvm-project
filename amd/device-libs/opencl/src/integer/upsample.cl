/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#define _C(X,Y) X##Y
#define C(X,Y) _C(X,Y)

#define ATTR __attribute__((overloadable, const))

#define char_shift 8
#define short_shift 16

#define char_up short
#define short_up int

#define GENN(N,T) \
ATTR C(T##_up,N) \
upsample(T##N hi, u##T##N lo) \
{ \
    return C(convert_,C(T##_up,N))((convert_uint##N(as_u##T##N(hi)) << T##_shift) | convert_uint##N(lo)); \
} \
 \
ATTR C(u,C(T##_up,N)) \
upsample(u##T##N hi, u##T##N lo) \
{ \
    return C(convert_u,C(T##_up,N))((convert_uint##N(hi) << T##_shift) | convert_uint##N(lo)); \
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

#define LGEN(N) \
ATTR long##N \
upsample(int##N hi, uint##N lo) \
{ \
    return as_long##N((convert_ulong##N(as_uint##N(hi)) << 32) | convert_ulong##N(lo)); \
} \
 \
ATTR ulong##N \
upsample(uint##N hi, uint##N lo) \
{ \
    return (convert_ulong##N(hi) << 32) | convert_ulong##N(lo); \
}

LGEN(16)
LGEN(8)
LGEN(4)
LGEN(3)
LGEN(2)
LGEN()

