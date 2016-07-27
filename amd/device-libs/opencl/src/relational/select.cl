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

#define char_mask ((char)1 << 7)
#define short_mask ((short)1 << 15)
#define int_mask ((int)1 << 31)
#define long_mask ((long)1 << 63)
#define float_mask int_mask
#define double_mask long_mask
#define half_mask short_mask

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

#define FGENS(T) \
IATTR static T \
sels_##T(T a, T b, T##_itype c) \
{ \
    return c ? b : a; \
} \
extern AATTR(S(sels_##T)) T select(T, T, T##_itype); \
extern AATTR(S(sels_##T)) T select(T, T, T##_utype); 

#define FGENV(N,T) \
IATTR static T##N \
selv_##T##N(T##N a, T##N b, C(T##_itype,N) c) \
{ \
    return as_##T##N(bitselect(C(as_,C(T##_itype,N))(a), C(as_,C(T##_itype,N))(b), (c & (C(T##_itype,N))T##_mask) != (C(T##_itype,N))0)); \
} \
extern AATTR(S(selv_##T##N)) T##N select(T##N, T##N, C(T##_itype,N)); \
extern AATTR(S(selv_##T##N)) T##N select(T##N, T##N, C(T##_utype,N));

#define FGEN(T) \
    FGENV(16,T) \
    FGENV(8,T) \
    FGENV(4,T) \
    FGENV(3,T) \
    FGENV(2,T) \
    FGENS(T)

FGEN(float)
FGEN(double)
FGEN(half)

#define IGENS(T) \
IATTR static T \
sels_##T(T a, T b, T c) \
{ \
    return c ? b : a; \
} \
extern AATTR(S(sels_##T)) T select(T, T, T); \
extern AATTR(S(sels_##T)) T select(T, T, T##_utype); \
extern AATTR(S(sels_##T)) T##_utype select(T##_utype, T##_utype, T); \
extern AATTR(S(sels_##T)) T##_utype select(T##_utype, T##_utype, T##_utype); 

#define IGENV(N,T) \
IATTR static T##N \
selv_##T##N(T##N a, T##N b, T##N c) \
{ \
    return bitselect(a, b, (c & (T##N)T##_mask) != (T##N)0); \
} \
extern AATTR(S(selv_##T##N)) T##N select(T##N, T##N, T##N); \
extern AATTR(S(selv_##T##N)) T##N select(T##N, T##N, C(T##_utype,N)); \
extern AATTR(S(selv_##T##N)) C(T##_utype,N) select(C(T##_utype,N), C(T##_utype,N), T##N); \
extern AATTR(S(selv_##T##N)) C(T##_utype,N) select(C(T##_utype,N), C(T##_utype,N), C(T##_utype,N));

#define IGEN(T) \
    IGENV(16,T) \
    IGENV(8,T) \
    IGENV(4,T) \
    IGENV(3,T) \
    IGENV(2,T) \
    IGENS(T)

IGEN(char)
IGEN(short)
IGEN(int)
IGEN(long)

