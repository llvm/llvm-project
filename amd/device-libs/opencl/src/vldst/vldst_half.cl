/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#define LATTR __attribute__((overloadable, pure))
#define SATTR __attribute__((overloadable))

#define LGENAN(N,A) \
LATTR float##N \
vload_half##N(size_t i, const A half *p) \
{ \
    return convert_float##N(vload##N(i, p)); \
}

#define LGENA1(A) \
LATTR float \
vload_half(size_t i, const A half *p) \
{ \
    return convert_float(p[i]); \
}

#define LGENA(A) \
    LGENAN(16,A) \
    LGENAN(8,A) \
    LGENAN(4,A) \
    LGENAN(3,A) \
    LGENAN(2,A) \
    LGENA1(A)

LGENA(__constant)
LGENA(__global)
LGENA(__local)
LGENA(__private)
LGENA()

#define LAGENAN(N,A) \
LATTR float##N \
vloada_half##N(size_t i, const A half *p) \
{ \
    return convert_float##N(*(const A half##N *)(p + i*N)); \
}

#define LAGENA3(A) \
LATTR float3 \
vloada_half3(size_t i, const A half *p) \
{ \
    half4 v = *(const A half4 *)(p + i*4); \
    return convert_float3(v.s012); \
}

#define LAGENA1(A) \
LATTR float \
vloada_half(size_t i, const A half *p) \
{ \
    return convert_float(p[i]); \
}

#define LAGENA(A) \
    LAGENAN(16,A) \
    LAGENAN(8,A) \
    LAGENAN(4,A) \
    LAGENA3(A) \
    LAGENAN(2,A) \
    LAGENA1(A)

LAGENA(__constant)
LAGENA(__global)
LAGENA(__local)
LAGENA(__private)
LAGENA()

#define SGENTARN(N,T,A,R) \
SATTR void \
vstore_half##N##R(T##N v, size_t i, A half *p) \
{ \
    vstore##N(convert_half##N##R(v), i, p); \
}

#define SGENTAR1(T,A,R) \
SATTR void \
vstore_half##R(T v, size_t i, A half *p) \
{ \
    p[i] = convert_half##R(v); \
}

#define SGENTAR(T,A,R) \
    SGENTARN(16,T,A,R) \
    SGENTARN(8,T,A,R) \
    SGENTARN(4,T,A,R) \
    SGENTARN(3,T,A,R) \
    SGENTARN(2,T,A,R) \
    SGENTAR1(T,A,R)

#define SGENTA(T,A) \
    SGENTAR(T,A,) \
    SGENTAR(T,A,_rte) \
    SGENTAR(T,A,_rtn) \
    SGENTAR(T,A,_rtp) \
    SGENTAR(T,A,_rtz)

#define SGENT(T) \
    SGENTA(T,__global) \
    SGENTA(T,__local) \
    SGENTA(T,__private) \
    SGENTA(T,)

SGENT(float)
SGENT(double)

#define SAGENTARN(N,T,A,R) \
SATTR void \
vstorea_half##N##R(T##N v, size_t i, A half *p) \
{ \
    *(A half##N *)(p + i*N) = convert_half##N##R(v); \
}

#define SAGENTAR3(T,A,R) \
SATTR void \
vstorea_half3##R(T##3 v, size_t i, A half *p) \
{ \
    half4 h; \
    h.s012 = convert_half3##R(v); \
    *(A half4 *)(p + i*4) = h; \
}

#define SAGENTAR1(T,A,R) \
SATTR void \
vstorea_half##R(T v, size_t i, A half *p) \
{ \
    p[i] = convert_half##R(v); \
}

#define SAGENTAR(T,A,R) \
    SAGENTARN(16,T,A,R) \
    SAGENTARN(8,T,A,R) \
    SAGENTARN(4,T,A,R) \
    SAGENTAR3(T,A,R) \
    SAGENTARN(2,T,A,R) \
    SAGENTAR1(T,A,R)

#define SAGENTA(T,A) \
    SAGENTAR(T,A,) \
    SAGENTAR(T,A,_rte) \
    SAGENTAR(T,A,_rtn) \
    SAGENTAR(T,A,_rtp) \
    SAGENTAR(T,A,_rtz)

#define SAGENT(T) \
    SAGENTA(T,__global) \
    SAGENTA(T,__local) \
    SAGENTA(T,__private) \
    SAGENTA(T,)

SAGENT(float)
SAGENT(double)

