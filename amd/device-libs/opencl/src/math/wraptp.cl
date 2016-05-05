
#include "ocml.h"

#pragma OPENCL EXTENSION cl_khr_fp16 : enable

#define _C(X,Y) X##Y
#define C(X,Y) _C(X,Y)

#define ATTR __attribute__((always_inline, overloadable))

#define float_suf _f32
#define double_suf _f64
#define half_suf _f16

#define ONAME(F,T) C(__ocml_,C(F,T##_suf))

#define EVN(N,F,T,P) \
    P v##N; \
    T r##N = ONAME(F,T)(x.s##N, y.s##N, &v##N)

#define EVAL2(F,T,P) EVN(0,F,T,P); EVN(1,F,T,P)
#define EVAL3(F,T,P) EVAL2(F,T,P); EVN(2,F,T,P)
#define EVAL4(F,T,P) EVAL2(F,T,P); EVN(2,F,T,P); EVN(3,F,T,P)
#define EVAL8(F,T,P) EVAL4(F,T,P); EVN(4,F,T,P); EVN(5,F,T,P); EVN(6,F,T,P); EVN(7,F,T,P)
#define EVAL16(F,T,P) EVAL8(F,T,P); EVN(8,F,T,P); EVN(9,F,T,P); EVN(a,F,T,P); EVN(b,F,T,P); EVN(c,F,T,P); EVN(d,F,T,P); EVN(e,F,T,P); EVN(f,F,T,P)

#define LIST2(V) V##0, V##1
#define LIST3(V) V##0, V##1, V##2
#define LIST4(V) LIST2(V), V##2, V##3
#define LIST8(V) LIST4(V), V##4, V##5, V##6, V##7
#define LIST16(V) LIST8(V), V##8, V##9, V##a, V##b, V##c, V##d, V##e, V##f

#define WRAPNTAP(N,F,T,A,P) \
ATTR T##N \
F(T##N x, T##N y, A P##N * v) \
{ \
    EVAL##N(F,T,P); \
    *v = (P##N)( LIST##N(v) ); \
    return (T##N) ( LIST##N(r) ); \
}

#define WRAP1TAP(F,T,A,P) \
ATTR T \
F(T x, T y, A P * v) \
{ \
    P v0; \
    T r0 = ONAME(F,T)(x, y, &v0); \
    *v = v0; \
    return r0; \
}

#define WRAPTAP(F,T,A,P) \
    WRAPNTAP(16,F,T,A,P) \
    WRAPNTAP(8,F,T,A,P) \
    WRAPNTAP(4,F,T,A,P) \
    WRAPNTAP(3,F,T,A,P) \
    WRAPNTAP(2,F,T,A,P) \
    WRAP1TAP(F,T,A,P)

WRAPTAP(remquo,float,,int)
WRAPTAP(remquo,double,,int)
WRAPTAP(remquo,half,,int)

