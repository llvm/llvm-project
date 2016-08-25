

#include "wg.h"

#define uint_ld(P) __llvm_ld_atomic_a3_x_wg_i32(P)
#define int_ld(P) as_int(__llvm_ld_atomic_a3_x_wg_i32((__local uint *)P))
#define ulong_ld(P) __llvm_ld_atomic_a3_x_wg_i64(P)
#define long_ld(P) as_long(__llvm_ld_atomic_a3_x_wg_i64((__local ulong *)P))

#define uint_st(P,X) __llvm_st_atomic_a3_x_wg_i32(P,X)
#define int_st(P,X) __llvm_st_atomic_a3_x_wg_i32((__local uint *)P, as_uint(X))
#define ulong_st(P,X) __llvm_st_atomic_a3_x_wg_i64(P,X)
#define long_st(P,X) __llvm_st_atomic_a3_x_wg_i64((__local ulong *)P, as_ulong(X))

#define uint_add(P,X) __llvm_atomic_add_a3_x_wg_i32(P,X)
#define int_add(P,X) as_int(__llvm_atomic_add_a3_x_wg_i32((__local uint *)P, as_uint(X)))
#define ulong_add(P,X) __llvm_atomic_add_a3_x_wg_i64(P,X)
#define long_add(P,X) as_long(__llvm_atomic_add_a3_x_wg_i64((__local ulong *)P, as_ulong(X)))

#define uint_max __llvm_atomic_umax_a3_x_wg_i32
#define int_max  __llvm_atomic_max_a3_x_wg_i32
#define ulong_max __llvm_atomic_umax_a3_x_wg_i64
#define long_max  __llvm_atomic_max_a3_x_wg_i64

#define uint_min __llvm_atomic_umin_a3_x_wg_i32
#define int_min  __llvm_atomic_min_a3_x_wg_i32
#define ulong_min __llvm_atomic_umin_a3_x_wg_i64
#define long_min  __llvm_atomic_min_a3_x_wg_i64

#define AGEN(TYPE,OP) \
__attribute__((overloadable, always_inline)) TYPE \
work_group_reduce_##OP(TYPE a) \
{ \
    uint n = get_num_sub_groups(); \
    a = sub_group_reduce_##OP(a); \
    if (n == 1) \
        return a; \
 \
    __local TYPE *p = (__local TYPE *)__get_scratch_lds(); \
    uint l = get_sub_group_local_id(); \
    uint i = get_sub_group_id(); \
 \
    if ((i == 0) & (l == 0)) \
        TYPE##_st(p, a); \
 \
    work_group_barrier(CLK_LOCAL_MEM_FENCE); \
    if ((i != 0) & (l == 0)) \
        TYPE##_##OP(p, a); \
    work_group_barrier(CLK_LOCAL_MEM_FENCE); \
    a = TYPE##_ld(p); \
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

#define SGEN(TYPE,OP,ID) \
__attribute__((overloadable, always_inline)) TYPE \
work_group_reduce_##OP(TYPE a) \
{ \
    uint n = get_num_sub_groups(); \
    a = sub_group_reduce_##OP(a); \
    if (n == 1) \
        return a; \
 \
    __local TYPE *p = (__local TYPE *)__get_scratch_lds(); \
    uint l = get_sub_group_local_id(); \
    uint i = get_sub_group_id(); \
 \
    if (l == 0) \
	p[i] = a; \
 \
    work_group_barrier(CLK_LOCAL_MEM_FENCE); \
    if (i == 0) { \
	TYPE t = l < n ? p[l] : ID; \
	t = sub_group_reduce_##OP(t); \
	if (l == 0) \
	    p[0] = t; \
    } \
    work_group_barrier(CLK_LOCAL_MEM_FENCE); \
    TYPE ret = p[0]; \
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

