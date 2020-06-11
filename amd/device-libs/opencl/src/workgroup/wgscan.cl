/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "wgscratch.h"

#pragma OPENCL EXTENSION cl_khr_fp16 : enable

// TODO Use a special scan for per-sub-group results since there
// are fewer of them than work-items in a sub group

#define add(X,Y) (X + Y)

#define GENI(TYPE,OP,ID) \
__attribute__((overloadable)) TYPE \
work_group_scan_inclusive_##OP(TYPE a) \
{ \
    uint n = get_num_sub_groups(); \
    a = sub_group_scan_inclusive_##OP(a); \
    if (n == 1) \
        return a; \
 \
    __local TYPE *p = (__local TYPE *)__get_scratch_lds(); \
    uint l = get_sub_group_local_id(); \
    uint i = get_sub_group_id(); \
 \
    if (l == get_sub_group_size() - 1U) \
	p[i] = a; \
 \
    work_group_barrier(CLK_LOCAL_MEM_FENCE); \
    if (i == 0) { \
	TYPE t = l < n ? p[l] : ID; \
	t = sub_group_scan_inclusive_##OP(t); \
	if (l < n) \
	    p[l] = t; \
    } \
    work_group_barrier(CLK_LOCAL_MEM_FENCE); \
    TYPE ret = i == 0 ? a : OP(a, p[i-1]); \
    work_group_barrier(CLK_LOCAL_MEM_FENCE); \
    return ret; \
}

GENI(int,add,0)
GENI(int,max,INT_MIN)
GENI(int,min,INT_MAX)

GENI(uint,add,0U)
GENI(uint,max,0U)
GENI(uint,min,UINT_MAX)

GENI(long,add,0L)
GENI(long,max,LONG_MIN)
GENI(long,min,LONG_MAX)

GENI(ulong,add,0UL)
GENI(ulong,max,0UL)
GENI(ulong,min,ULONG_MAX)

GENI(float,add,0.0f)
GENI(float,max,-INFINITY)
GENI(float,min,INFINITY)

GENI(double,add,0.0)
GENI(double,max,-(double)INFINITY)
GENI(double,min,(double)INFINITY)

GENI(half,add,0.0h)
GENI(half,max,-(half)INFINITY)
GENI(half,min,(half)INFINITY)

#define GENE(TYPE,OP,ID) \
__attribute__((overloadable)) TYPE \
work_group_scan_exclusive_##OP(TYPE a) \
{ \
    uint n = get_num_sub_groups(); \
    TYPE t = sub_group_scan_exclusive_##OP(a); \
    if (n == 1) \
        return t; \
 \
    __local TYPE *p = (__local TYPE *)__get_scratch_lds(); \
    uint l = get_sub_group_local_id(); \
    uint i = get_sub_group_id(); \
 \
    if (l == get_sub_group_size() - 1U) \
	p[i] = OP(a, t); \
 \
    work_group_barrier(CLK_LOCAL_MEM_FENCE); \
    if (i == 0) { \
	TYPE s = l < n ? p[l] : ID; \
	s = sub_group_scan_inclusive_##OP(s); \
	if (l < n) \
	    p[l] = s; \
    } \
    work_group_barrier(CLK_LOCAL_MEM_FENCE); \
    TYPE ret = i == 0 ? t : OP(t, p[i-1]); \
    work_group_barrier(CLK_LOCAL_MEM_FENCE); \
    return ret; \
}

GENE(int,add,0)
GENE(int,max,INT_MIN)
GENE(int,min,INT_MAX)

GENE(uint,add,0U)
GENE(uint,max,0U)
GENE(uint,min,UINT_MAX)

GENE(long,add,0L)
GENE(long,max,LONG_MIN)
GENE(long,min,LONG_MAX)

GENE(ulong,add,0UL)
GENE(ulong,max,0UL)
GENE(ulong,min,ULONG_MAX)

GENE(float,add,0.0f)
GENE(float,max,-INFINITY)
GENE(float,min,INFINITY)

GENE(double,add,0.0)
GENE(double,max,-(double)INFINITY)
GENE(double,min,(double)INFINITY)

GENE(half,add,0.0h)
GENE(half,max,-(half)INFINITY)
GENE(half,min,(half)INFINITY)

