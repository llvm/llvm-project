/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#define ATTR __attribute__((overloadable, const))

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

// On the signed implementation, we intentionally use unsigned integers to
// avoid signed integer overflows, which result in undefined-behaviour
#define LGENN(N,T) \
ATTR u##T##N \
abs_diff(T##N x, T##N y) \
{ \
    T##N c = x > y; \
    u##T##N xx = convert_u##T##N(x); \
    u##T##N yy = convert_u##T##N(y); \
    u##T##N xmy = xx - yy; \
    u##T##N ymx = yy - xx; \
    return select(ymx, xmy, c); \
} \
 \
ATTR u##T##N \
abs_diff(u##T##N x, u##T##N y) \
{ \
    T##N c = x > y; \
    u##T##N xmy = x - y; \
    u##T##N ymx = y - x; \
    return select(ymx, xmy, c); \
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
