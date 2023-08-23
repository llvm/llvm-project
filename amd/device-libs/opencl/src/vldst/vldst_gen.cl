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

#define LATTR __attribute__((overloadable, pure))
#define SATTR __attribute__((overloadable))

#define char_align 1
#define uchar_align 1
#define short_align 2
#define ushort_align 2
#define int_align 4
#define uint_align 4
#define long_align 8
#define ulong_align 8
#define float_align 4
#define double_align 8
#define half_align 2

#define LGENAN(N,A,T) \
LATTR T##N \
vload##N(size_t i, const A T *p) \
{ \
    typedef T __attribute__((ext_vector_type(N), aligned(T##_align))) vt; \
    p += i * N; \
    return *(const A vt *)p; \
}

#define LGENA3(A,T) \
LATTR T##3 \
vload3(size_t i, const A T *p) \
{ \
    p += i * 3; \
    return (T##3) ( p[0], p[1], p[2] ); \
}

#define LGENA(A,T) \
    LGENAN(16,A,T) \
    LGENAN(8,A,T) \
    LGENAN(4,A,T) \
    LGENA3(A,T) \
    LGENAN(2,A,T)

#define LGEN(T) \
    LGENA(__constant,T) \
    LGENA(__private,T) \
    LGENA(__local,T) \
    LGENA(__global,T) \
    LGENA(,T)

LGEN(char)
LGEN(uchar)
LGEN(short)
LGEN(ushort)
LGEN(int)
LGEN(uint)
LGEN(long)
LGEN(ulong)
LGEN(float)
LGEN(double)
LGEN(half)

#define SGENAN(N,A,T) \
SATTR void \
vstore##N(T##N v, size_t i, A T *p) \
{ \
    typedef T __attribute__((ext_vector_type(N), aligned(T##_align))) vt; \
    p += i * N; \
    *(A vt *)p = v; \
}

#define SGENA3(A,T) \
SATTR void \
vstore3(T##3 v, size_t i, A T *p) \
{ \
    p += i * 3; \
    p[0] = v.s0; \
    p[1] = v.s1; \
    p[2] = v.s2; \
}

#define SGENA(A,T) \
    SGENAN(16,A,T) \
    SGENAN(8,A,T) \
    SGENAN(4,A,T) \
    SGENA3(A,T) \
    SGENAN(2,A,T)

#define SGEN(T) \
    SGENA(__private,T) \
    SGENA(__local,T) \
    SGENA(__global,T) \
    SGENA(,T)

SGEN(char)
SGEN(uchar)
SGEN(short)
SGEN(ushort)
SGEN(int)
SGEN(uint)
SGEN(long)
SGEN(ulong)
SGEN(float)
SGEN(double)
SGEN(half)

