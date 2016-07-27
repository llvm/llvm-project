/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#define ATTR __attribute__((always_inline, overloadable, const))

#define GENN(N,T) \
ATTR u##T##N \
abs_diff(T##N x, T##N y) \
{ \
    int##N xx = convert_int##N(x); \
    int##N yy = convert_int##N(y); \
    int##N d = max(xx,yy) - min(xx,yy); \
    return convert_u##T##N(d); \
} \
 \
ATTR u##T##N \
abs_diff(u##T##N x, u##T##N y) \
{ \
    uint##N xx = convert_uint##N(x); \
    uint##N yy = convert_uint##N(y); \
    uint##N d = max(xx,yy) - min(xx,yy); \
    return convert_u##T##N(d); \
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
GEN(int)

#define LGEN(N) \
ATTR ulong##N \
abs_diff(long##N x, long##N y) \
{ \
    return as_ulong##N(select(y - x, x - y, x > y)); \
} \
 \
ATTR ulong##N \
abs_diff(ulong##N x, ulong##N y) \
{ \
    return select(y - x, x - y, x > y); \
}

LGEN(16)
LGEN(8)
LGEN(4)
LGEN(3)
LGEN(2)

ATTR ulong
abs_diff(long x, long y)
{
    long xmy = x - y;
    long ymx = y - x;
    return x > y ? xmy : ymx;
}

ATTR ulong
abs_diff(ulong x, ulong y)
{
    ulong xmy = x - y;
    ulong ymx = y - x;
    return x > y ? xmy : ymx;
}

