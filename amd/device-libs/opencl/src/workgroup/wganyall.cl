
#include "wg.h"

#define update_all atomic_fetch_and_explicit
#define update_any atomic_fetch_or_explicit

#define GEN_AA(SUF,ID) \
__attribute__((overloadable, always_inline)) int \
work_group_##SUF(int predicate) \
{ \
    uint n = get_num_sub_groups(); \
    int a = sub_group_##SUF(predicate); \
    if (n == 1) \
	return a; \
 \
    __local atomic_int *p = (__local atomic_int *)__get_scratch_lds(); \
    uint l = get_sub_group_local_id(); \
    uint i = get_sub_group_id(); \
 \
    if ((i == 0) & (l == 0)) \
	atomic_store_explicit(p, a, memory_order_relaxed, memory_scope_work_group); \
 \
    work_group_barrier(CLK_LOCAL_MEM_FENCE); \
    if ((i != 0) & (l == 0)) \
        update_##SUF(p, a, memory_order_relaxed, memory_scope_work_group); \
    work_group_barrier(CLK_LOCAL_MEM_FENCE); \
    a = atomic_load_explicit(p, memory_order_relaxed, memory_scope_work_group); \
    work_group_barrier(CLK_LOCAL_MEM_FENCE); \
 \
    return a; \
}

GEN_AA(all, 1U)
GEN_AA(any, 0U);

