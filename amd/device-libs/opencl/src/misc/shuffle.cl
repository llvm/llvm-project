/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#pragma OPENCL EXTENSION cl_khr_fp16 : enable

#define _S(X) #X
#define S(X) _S(X)

#define _C(X,Y) X##Y
#define C(X,Y) _C(X,Y)

#define char_utype uchar
#define short_utype ushort
#define int_utype uint
#define long_utype ulong
#define float_utype uint
#define double_utype ulong
#define half_utype ushort

#define ATTR __attribute__((always_inline, overloadable, const))
#define IATTR __attribute__((always_inline, const))
#define AATTR(A) __attribute__((always_inline, overloadable, const, alias(A)))

#define LIST2 t[m.s0], t[m.s1]
#define LIST4 LIST2, t[m.s2], t[m.s3]
#define LIST8 LIST4, t[m.s4], t[m.s5], t[m.s6], t[m.s7]
#define LIST16 LIST8, t[m.s8], t[m.s9], t[m.sa], t[m.sb], t[m.sc], t[m.sd], t[m.se], t[m.sf]

#define GENIMN(M,N,T) \
IATTR T##N \
sh_##N##T##M(T##M x, C(T##_utype,N) m) \
{ \
    __attribute__((aligned(sizeof(T##M)))) T t[M]; \
    *(__private T##M *)t = x; \
    m &= (C(T##_utype,N))(M-1); \
    return (T##N) ( LIST##N ); \
} \
extern AATTR(S(sh_##N##T##M)) T##N shuffle(T##M, C(T##_utype,N)); \
extern AATTR(S(sh_##N##T##M)) u##T##N shuffle(u##T##M, C(T##_utype,N)); \
 \
IATTR T##N \
sh2_##N##T##M(T##M x, T##M y, C(T##_utype,N) m) \
{ \
    __attribute__((aligned(sizeof(T##M)))) T t[2*M]; \
    *(__private T##M *)t = x; \
    *(__private T##M *)(t + M) = y; \
    m &= (C(T##_utype,N))(2*M-1); \
    return (T##N) ( LIST##N ); \
} \
extern AATTR(S(sh2_##N##T##M)) T##N shuffle2(T##M, T##M, C(T##_utype,N)); \
extern AATTR(S(sh2_##N##T##M)) u##T##N shuffle2(u##T##M, u##T##M, C(T##_utype,N));

#define GENIN(N,T) \
    GENIMN(16,N,T) \
    GENIMN(8,N,T) \
    GENIMN(4,N,T) \
    GENIMN(2,N,T)

#define GENI(T) \
    GENIN(16,T) \
    GENIN(8,T) \
    GENIN(4,T) \
    GENIN(2,T)

GENI(char)
GENI(short)
GENI(int)
GENI(long)

#define GENFMN(M,N,T) \
ATTR T##N \
shuffle(T##M x, C(T##_utype,N) m) \
{ \
    __attribute__((aligned(sizeof(T##M)))) T t[M]; \
    *(__private T##M *)t = x; \
    m &= (C(T##_utype,N))(M-1); \
    return (T##N) ( LIST##N ); \
} \
 \
ATTR T##N \
shuffle2(T##M x, T##M y, C(T##_utype,N) m) \
{ \
    __attribute__((aligned(sizeof(T##M)))) T t[2*M]; \
    *(__private T##M *)t = x; \
    *(__private T##M *)(t + M) = y; \
    m &= (C(T##_utype,N))(2*M-1); \
    return (T##N) ( LIST##N ); \
}

#define GENFN(N,T) \
    GENFMN(16,N,T) \
    GENFMN(8,N,T) \
    GENFMN(4,N,T) \
    GENFMN(2,N,T)

#define GENF(T) \
    GENFN(16,T) \
    GENFN(8,T) \
    GENFN(4,T) \
    GENFN(2,T)

GENF(float)
GENF(double)
GENF(half)

