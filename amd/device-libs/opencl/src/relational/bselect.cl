/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#define _S(X) #X
#define S(X) _S(X)

#define _C(A,B) A##B
#define C(A,B) _C(A,B)

#define ATTR __attribute__((always_inline, overloadable, const))
#define IATTR __attribute__((always_inline, const))
#define AATTR(S) __attribute__((always_inline, overloadable, const, alias(S)))

#pragma OPENCL EXTENSION cl_khr_fp16 : enable

#define char_utype uchar
#define short_utype ushort
#define int_utype uint
#define long_utype ulong
#define float_itype int
#define float_utype uint
#define double_itype long
#define double_utype ulong
#define half_itype short
#define half_utype ushort

#define FGENN(N,T) \
ATTR T##N \
bitselect(T##N a, T##N b, T##N c) \
{ \
    return as_##T##N(bitselect(C(as_,C(T##_itype,N))(a), C(as_,C(T##_itype,N))(b), C(as_,C(T##_itype,N))(c))); \
} \

#define FGEN(T) \
    FGENN(16,T) \
    FGENN(8,T) \
    FGENN(4,T) \
    FGENN(3,T) \
    FGENN(2,T) \
    FGENN(,T)

FGEN(float)
FGEN(double)
FGEN(half)

#define IGENN(N,T) \
IATTR static T##N \
bsel_##T##N(T##N a, T##N b, T##N c) \
{ \
    return a ^ ((a ^ b) & c); \
} \
extern AATTR(S(bsel_##T##N)) T##N bitselect(T##N, T##N, T##N); \
extern AATTR(S(bsel_##T##N)) C(T##_utype,N) bitselect(C(T##_utype,N), C(T##_utype,N), C(T##_utype,N));

#define IGEN(T) \
    IGENN(16,T) \
    IGENN(8,T) \
    IGENN(4,T) \
    IGENN(3,T) \
    IGENN(2,T) \
    IGENN(,T)

IGEN(char)
IGEN(short)
IGEN(int)
IGEN(long)

