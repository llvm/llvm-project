/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "ocml.h"

#pragma OPENCL EXTENSION cl_khr_fp16 : enable

#define _C(X,Y) X##Y
#define C(X,Y) _C(X,Y)

#define ATTR __attribute__((overloadable))

#define float_ssuf _f32
#define double_ssuf _f64
#define half_ssuf _f16
#define half_psuf _2f16

#define SNAME(F,T) C(__ocml_,C(F,T##_ssuf))
#define PNAME(F,T) C(__ocml_,C(F,T##_psuf))

#define SEVN(N,F,T,P) \
    P v##N; \
    T r##N = SNAME(F,T)(x.s##N, y.s##N, &v##N)

#define PEVN(N,F,T,P) \
    P##2 v##N; \
    T##2 r##N = PNAME(F,T)(x.s##N, y.s##N, &v##N)

#define SEVAL2(F,T,P) SEVN(0,F,T,P); SEVN(1,F,T,P)
#define SEVAL3(F,T,P) SEVAL2(F,T,P); SEVN(2,F,T,P)
#define SEVAL4(F,T,P) SEVAL2(F,T,P); SEVN(2,F,T,P); SEVN(3,F,T,P)
#define SEVAL8(F,T,P) SEVAL4(F,T,P); SEVN(4,F,T,P); SEVN(5,F,T,P); SEVN(6,F,T,P); SEVN(7,F,T,P)
#define SEVAL16(F,T,P) SEVAL8(F,T,P); SEVN(8,F,T,P); SEVN(9,F,T,P); SEVN(a,F,T,P); SEVN(b,F,T,P); SEVN(c,F,T,P); SEVN(d,F,T,P); SEVN(e,F,T,P); SEVN(f,F,T,P)

#define PEVAL3(F,T,P) PEVN(01,F,T,P); SEVN(2,F,T,P)
#define PEVAL4(F,T,P) PEVN(01,F,T,P); PEVN(23,F,T,P)
#define PEVAL8(F,T,P) PEVAL4(F,T,P); PEVN(45,F,T,P); PEVN(67,F,T,P)
#define PEVAL16(F,T,P) PEVAL8(F,T,P); PEVN(89,F,T,P); PEVN(ab,F,T,P); PEVN(cd,F,T,P); PEVN(ef,F,T,P)

#define SLST2(V) V##0, V##1
#define SLST3(V) SLST2(V), V##2
#define SLST4(V) SLST2(V), V##2, V##3
#define SLST8(V) SLST4(V), V##4, V##5, V##6, V##7
#define SLST16(V) SLST8(V), V##8, V##9, V##a, V##b, V##c, V##d, V##e, V##f

#define PLST3(V) V##01, V##2
#define PLST4(V) V##01, V##23
#define PLST8(V) PLST4(V), V##45, V##67
#define PLST16(V) PLST8(V), V##89, V##ab, V##cd, V##ef

#define SWRAPNTAP(N,F,T,A,P) \
ATTR T##N \
F(T##N x, T##N y, A P##N * v) \
{ \
    SEVAL##N(F,T,P); \
    *v = (P##N)( SLST##N(v) ); \
    return (T##N) ( SLST##N(r) ); \
}

#define PWRAPNTAP(N,F,T,A,P) \
ATTR T##N \
F(T##N x, T##N y, A P##N * v) \
{ \
    PEVAL##N(F,T,P); \
    *v = (P##N)( PLST##N(v) ); \
    return (T##N) ( PLST##N(r) ); \
}

#define WRAP1TAP(F,T,A,P) \
ATTR T \
F(T x, T y, A P * v) \
{ \
    P v0; \
    T r0 = SNAME(F,T)(x, y, &v0); \
    *v = v0; \
    return r0; \
}

#define WRAP2TAP(F,T,A,P) \
ATTR T##2 \
F(T##2 x, T##2 y, A P##2 * v) \
{ \
    P##2 v01; \
    T##2 r01 = PNAME(F,T)(x, y, &v01); \
    *v = v01; \
    return r01; \
}

#define SWRAPTAP(F,T,A,P) \
    SWRAPNTAP(16,F,T,A,P) \
    SWRAPNTAP(8,F,T,A,P) \
    SWRAPNTAP(4,F,T,A,P) \
    SWRAPNTAP(3,F,T,A,P) \
    SWRAPNTAP(2,F,T,A,P) \
    WRAP1TAP(F,T,A,P)

#define PWRAPTAP(F,T,A,P) \
    PWRAPNTAP(16,F,T,A,P) \
    PWRAPNTAP(8,F,T,A,P) \
    PWRAPNTAP(4,F,T,A,P) \
    PWRAPNTAP(3,F,T,A,P) \
    WRAP2TAP(F,T,A,P) \
    WRAP1TAP(F,T,A,P)

#define SWRAPTP(F,T,P) \
    SWRAPTAP(F,T,__private,P) \
    SWRAPTAP(F,T,__local,P) \
    SWRAPTAP(F,T,__global,P) \
    SWRAPTAP(F,T,,P)

#define PWRAPTP(F,T,P) \
    PWRAPTAP(F,T,__private,P) \
    PWRAPTAP(F,T,__local,P) \
    PWRAPTAP(F,T,__global,P) \
    PWRAPTAP(F,T,,P)

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated"

SWRAPTP(remquo,float,int)
SWRAPTP(remquo,double,int)
PWRAPTP(remquo,half,int)

#pragma clang diagnostic pop
