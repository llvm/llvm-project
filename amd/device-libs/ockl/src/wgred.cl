/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "oclc.h"
#include "ockl.h"
#include "wgscratch.h"

#define _C(X,Y) X##Y
#define C(X,Y) _C(X,Y)

#define reduce_add __opencl_atomic_fetch_add
#define reduce_and __opencl_atomic_fetch_and
#define reduce_or __opencl_atomic_fetch_or

#define int_suf _i32

static uint
my_num_sub_groups(void)
{
    uint wgs = __ockl_mul24_i32((uint)__ockl_get_local_size(2),
                                __ockl_mul24_i32((uint)__ockl_get_local_size(1),
                                                 (uint)__ockl_get_local_size(0)));
    return (wgs + OCLC_WAVEFRONT_SIZE - 1) >> __oclc_wavefrontsize_log2;
}

static uint
my_sub_group_id(void)
{
    return (uint)__ockl_get_local_linear_id() >> __oclc_wavefrontsize_log2;
}

static void
my_barrier(void)
{
    __builtin_amdgcn_fence(__ATOMIC_RELEASE, "workgroup");
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "workgroup");
}

#define AGEN(T,OP) \
T \
C(__ockl_wgred_,C(OP,T##_suf))(int a) \
{ \
    uint n = my_num_sub_groups(); \
    a = C(__ockl_wfred_##OP,T##_suf)(a); \
    if (n == 1) \
        return a; \
 \
    __local atomic_##T *p = (__local atomic_##T *)__get_scratch_lds(); \
    uint l = __ockl_lane_u32(); \
    uint i = my_sub_group_id(); \
 \
    if ((i == 0) & (l == 0)) \
        __opencl_atomic_store(p, a, memory_order_relaxed, memory_scope_work_group); \
 \
    my_barrier(); \
    if ((i != 0) & (l == 0)) \
        reduce_##OP(p, a, memory_order_relaxed, memory_scope_work_group); \
    my_barrier(); \
    a = __opencl_atomic_load(p, memory_order_relaxed, memory_scope_work_group); \
    my_barrier(); \
    return a; \
}

AGEN(int,add)
AGEN(int,and)
AGEN(int,or)
