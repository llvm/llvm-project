/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#pragma OPENCL EXTENSION cl_khr_fp16 : enable

#define _S(X) #X
#define S(X) _S(X)

#define ATTR __attribute__((overloadable))
#define IATTR
#define AATTR(A) __attribute__((overloadable, alias(A)))

#define BODY(D,S) \
    size_t i; \
    size_t d = mul24(mul24((int)get_local_size(0), (int)get_local_size(1)), (int)get_local_size(2)); \
    for (i = get_local_linear_id(); i<n; i += d) \
        dst[D] = src[S]; \
    return e


#define GENIN(N,T) \
IATTR static event_t \
gli_##T##N(__global T##N *dst, const __local T##N *src, size_t n, event_t e) \
{ \
    BODY(i,i); \
} \
extern AATTR(S(gli_##T##N)) event_t async_work_group_copy(__global u##T##N *, const __local u##T##N *, size_t, event_t); \
extern AATTR(S(gli_##T##N)) event_t async_work_group_copy(__global T##N *, const __local T##N *, size_t, event_t); \
 \
IATTR static event_t \
lgi_##T##N(__local T##N *dst, const __global T##N *src, size_t n, event_t e) \
{ \
    BODY(i,i); \
} \
extern AATTR(S(lgi_##T##N)) event_t async_work_group_copy(__local u##T##N *, const __global u##T##N *, size_t, event_t); \
extern AATTR(S(lgi_##T##N)) event_t async_work_group_copy(__local T##N *, const __global T##N *, size_t, event_t); \
 \
IATTR static event_t \
sgli_##T##N(__global T##N *dst, const __local T##N *src, size_t n, size_t j, event_t e) \
{ \
    BODY(i*j,i); \
} \
extern AATTR(S(sgli_##T##N)) event_t async_work_group_strided_copy(__global u##T##N *, const __local u##T##N *, size_t, size_t, event_t); \
extern AATTR(S(sgli_##T##N)) event_t async_work_group_strided_copy(__global T##N *, const __local T##N *, size_t, size_t, event_t); \
 \
IATTR static event_t \
slgi_##T##N(__local T##N *dst, const __global T##N *src, size_t n, size_t j, event_t e) \
{ \
    BODY(i,i*j); \
} \
extern AATTR(S(slgi_##T##N)) event_t async_work_group_strided_copy(__local u##T##N *, const __global u##T##N *, size_t, size_t, event_t); \
extern AATTR(S(slgi_##T##N)) event_t async_work_group_strided_copy(__local T##N *, const __global T##N *, size_t, size_t, event_t);

#define GENI(T) \
    GENIN(16,T) \
    GENIN(8,T) \
    GENIN(4,T) \
    GENIN(3,T) \
    GENIN(2,T) \
    GENIN(,T) \

GENI(char)
GENI(short)
GENI(int)
GENI(long)

#define GENFN(N,T) \
ATTR event_t \
async_work_group_copy(__global T##N *dst, const __local T##N *src, size_t n, event_t e) \
{ \
    BODY(i,i); \
} \
 \
ATTR event_t \
async_work_group_copy(__local T##N *dst, const __global T##N *src, size_t n, event_t e) \
{ \
    BODY(i,i); \
} \
 \
ATTR event_t \
async_work_group_strided_copy(__global T##N *dst, const __local T##N *src, size_t n, size_t j, event_t e) \
{ \
    BODY(i*j,i); \
} \
 \
ATTR event_t \
async_work_group_strided_copy(__local T##N *dst, const __global T##N *src, size_t n, size_t j, event_t e) \
{ \
    BODY(i,i*j); \
} \

#define GENF(T) \
    GENFN(16,T) \
    GENFN(8,T) \
    GENFN(4,T) \
    GENFN(3,T) \
    GENFN(2,T) \
    GENFN(,T) \

GENF(float)
GENF(double)
GENF(half)

