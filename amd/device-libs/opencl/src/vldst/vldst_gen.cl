/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#pragma OPENCL EXTENSION cl_khr_fp16 : enable

#define _C(X,Y) X##Y
#define C(X,Y) _C(X,Y)

#define _S(X) #X
#define S(X) _S(X)

#define LATTR __attribute__((always_inline, overloadable, pure))
#define SATTR __attribute__((always_inline, overloadable))
#define ILATTR __attribute__((always_inline, pure))
#define ISATTR __attribute__((always_inline))
#define ALATTR(A) __attribute__((always_inline, overloadable, pure, alias(A)))
#define ASATTR(A) __attribute__((always_inline, overloadable, alias(A)))

#define char_align 1
#define short_align 2
#define int_align 4
#define long_align 8

#define float_itype int
#define double_itype long
#define half_itype short

#define char_utype uchar
#define short_utype ushort
#define int_utype uint
#define long_utype ulong

#define __constant_pfx c
#define __global_pfx g
#define __local_pfx l
#define __private_pfx p
#define _pfx

#define FLGENAN(N,A,T) \
LATTR T##N \
vload##N(size_t i, const A T *p) \
{ \
    return as_##T##N(vload##N(i, (const A T##_itype *)p)); \
}

#define FLGENA(A,T) \
    FLGENAN(16,A,T) \
    FLGENAN(8,A,T) \
    FLGENAN(4,A,T) \
    FLGENAN(3,A,T) \
    FLGENAN(2,A,T)

#define FLGEN(T) \
    FLGENA(__constant,T) \
    FLGENA(,T)

FLGEN(float)
FLGEN(double)
FLGEN(half)

#define FSGENAN(N,A,T) \
SATTR void \
vstore##N(T##N v, size_t i, A T *p) \
{ \
    vstore##N(C(as_,C(T##_itype,N))(v), i, (A T##_itype *)p); \
}

#define FSGENA(A,T) \
    FSGENAN(16,A,T) \
    FSGENAN(8,A,T) \
    FSGENAN(4,A,T) \
    FSGENAN(3,A,T) \
    FSGENAN(2,A,T)

#define FSGEN(T) \
    FSGENA(,T)

FSGEN(float)
FSGEN(double)
FSGEN(half)

#define ILGENAN(N,A,T) \
ILATTR static T##N \
C(A##_pfx,vld_##T##N)(size_t i, const A T *p) \
{ \
    typedef T __attribute__((ext_vector_type(N), aligned(T##_align))) vt; \
    p += i * N; \
    return *(const A vt *)p; \
} \
extern ALATTR(S(C(A##_pfx,vld_##T##N))) T##N vload##N(size_t, const A T *); \
extern ALATTR(S(C(A##_pfx,vld_##T##N))) C(T##_utype,N) vload##N(size_t, const A T##_utype *);

#define ILGENA3(A,T) \
ILATTR static T##3 \
C(A##_pfx,vld_##T##3)(size_t i, const A T *p) \
{ \
    p += i * 3; \
    return (T##3) ( p[0], p[1], p[2] ); \
} \
extern ALATTR(S(C(A##_pfx,vld_##T##3))) T##3 vload##3(size_t, const A T *); \
extern ALATTR(S(C(A##_pfx,vld_##T##3))) C(T##_utype,3) vload##3(size_t, const A T##_utype *);

#define ILGENA(A,T) \
    ILGENAN(16,A,T) \
    ILGENAN(8,A,T) \
    ILGENAN(4,A,T) \
    ILGENA3(A,T) \
    ILGENAN(2,A,T)

#define ILGEN(T) \
    ILGENA(__constant,T) \
    ILGENA(,T)

ILGEN(char)
ILGEN(short)
ILGEN(int)
ILGEN(long)


#define ISGENAN(N,A,T) \
ISATTR static void \
C(A##_pfx,vst_##T##N)(T##N v, size_t i, A T *p) \
{ \
    typedef T __attribute__((ext_vector_type(N), aligned(T##_align))) vt; \
    p += i * N; \
    *(A vt *)p = v; \
} \
extern ASATTR(S(C(A##_pfx,vst_##T##N))) void vstore##N(T##N, size_t, A T *); \
extern ASATTR(S(C(A##_pfx,vst_##T##N))) void vstore##N(C(T##_utype,N), size_t, A T##_utype *);

#define ISGENA3(A,T) \
ISATTR static void \
C(A##_pfx,vst_##T##3)(T##3 v, size_t i, A T *p) \
{ \
    p += i * 3; \
    p[0] = v.s0; \
    p[1] = v.s1; \
    p[2] = v.s2; \
} \
extern ASATTR(S(C(A##_pfx,vst_##T##3))) void vstore##3(T##3, size_t, A T *); \
extern ASATTR(S(C(A##_pfx,vst_##T##3))) void vstore##3(C(T##_utype,3), size_t, A T##_utype *);

#define ISGENA(A,T) \
    ISGENAN(16,A,T) \
    ISGENAN(8,A,T) \
    ISGENAN(4,A,T) \
    ISGENA3(A,T) \
    ISGENAN(2,A,T)

#define ISGEN(T) \
    ISGENA(,T)

ISGEN(char)
ISGEN(short)
ISGEN(int)
ISGEN(long)

