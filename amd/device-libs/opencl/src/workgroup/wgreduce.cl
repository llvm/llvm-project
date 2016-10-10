/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "wg.h"

#define reduce_add atomic_fetch_add_explicit
#define reduce_min atomic_fetch_min_explicit
#define reduce_max atomic_fetch_max_explicit

#define AGEN(T,OP) \
__attribute__((overloadable, always_inline)) T \
work_group_reduce_##OP(T a) \
{ \
    uint n = get_num_sub_groups(); \
    a = sub_group_reduce_##OP(a); \
    if (n == 1) \
        return a; \
 \
    __local atomic_##T *p = (__local atomic_##T *)__get_scratch_lds(); \
    uint l = get_sub_group_local_id(); \
    uint i = get_sub_group_id(); \
 \
    if ((i == 0) & (l == 0)) \
        atomic_store_explicit(p, a, memory_order_relaxed, memory_scope_work_group); \
 \
    work_group_barrier(CLK_LOCAL_MEM_FENCE); \
    if ((i != 0) & (l == 0)) \
        reduce_##OP(p, a, memory_order_relaxed, memory_scope_work_group); \
    work_group_barrier(CLK_LOCAL_MEM_FENCE); \
    a = atomic_load_explicit(p, memory_order_relaxed, memory_scope_work_group); \
    work_group_barrier(CLK_LOCAL_MEM_FENCE); \
    return a; \
}

AGEN(int,add)
AGEN(int,max)
AGEN(int,min)

AGEN(uint,add)
AGEN(uint,max)
AGEN(uint,min)

AGEN(long,add)
AGEN(long,max)
AGEN(long,min)

AGEN(ulong,add)
AGEN(ulong,max)
AGEN(ulong,min)

// TODO implement floating point reduction using LDS atomics as above
//      (note that ds_add_f32 is not available on GFX7)

// TODO Use a special reduce for per-sub-group results since there
// are fewer of them than work-items in a sub group

#define add(X,Y) (X + Y)

#define SGEN(T,OP,ID) \
__attribute__((overloadable, always_inline)) T \
work_group_reduce_##OP(T a) \
{ \
    uint n = get_num_sub_groups(); \
    a = sub_group_reduce_##OP(a); \
    if (n == 1) \
        return a; \
 \
    __local T *p = (__local T *)__get_scratch_lds(); \
    uint l = get_sub_group_local_id(); \
    uint i = get_sub_group_id(); \
 \
    if (l == 0) \
	p[i] = a; \
 \
    work_group_barrier(CLK_LOCAL_MEM_FENCE); \
    if (i == 0) { \
	T t = l < n ? p[l] : ID; \
	t = sub_group_reduce_##OP(t); \
	if (l == 0) \
	    p[0] = t; \
    } \
    work_group_barrier(CLK_LOCAL_MEM_FENCE); \
    T ret = p[0]; \
    work_group_barrier(CLK_LOCAL_MEM_FENCE); \
    return ret; \
}

SGEN(float,add,0.0f)
SGEN(float,max,-INFINITY)
SGEN(float,min,INFINITY)

SGEN(double,add,0.0)
SGEN(double,max,-(double)INFINITY)
SGEN(double,min,(double)INFINITY)

SGEN(half,add,0.0h)
SGEN(half,max,-(half)INFINITY)
SGEN(half,min,(half)INFINITY)

