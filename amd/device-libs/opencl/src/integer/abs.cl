/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#define ATTR __attribute__((always_inline, overloadable, const))

#define GENN(N,T) \
ATTR u##T##N \
abs(T##N x) \
{ \
    int##N px = convert_int##N(x); \
    int##N nx = -px; \
    return convert_u##T##N(max(px,nx)); \
} \
 \
ATTR u##T##N \
abs(u##T##N x) \
{ \
    return x; \
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
ATTR u##T##N \
abs(T##N x) \
{ \
    return convert_u##T##N(select(-x, x, x > (T)0)); \
} \
 \
ATTR u##T##N \
abs(u##T##N x) \
{ \
    return x; \
}

#define LGEN1(T) \
ATTR u##T \
abs(T x) \
{ \
    T mx = -x; \
    return as_u##T(x > (T)0 ? x : mx); \
} \
 \
ATTR u##T \
abs(u##T x) \
{ \
    return x; \
}

#define LGEN(T) \
    LGENN(16,T) \
    LGENN(8,T) \
    LGENN(4,T) \
    LGENN(3,T) \
    LGENN(2,T) \
    LGEN1(T)

LGEN(int)
LGEN(long)

