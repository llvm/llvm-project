
#include "wg.h"

#define update_all __llvm_atomic_and_a3_x_wg_i32
#define update_any __llvm_atomic_or_a3_x_wg_i32

#define GEN_AA(SUF,ID) \
__attribute__((overloadable, always_inline)) int \
work_group_##SUF(int predicate) \
{ \
    uint n = get_num_sub_groups(); \
    int a = sub_group_##SUF(predicate); \
    if (n == 1) \
	return a; \
 \
    __local uint *p = (__local uint *)__get_scratch_lds(); \
    uint l = get_sub_group_local_id(); \
    uint i = get_sub_group_id(); \
 \
    if ((i == 0) & (l == 0)) \
        __llvm_st_atomic_a3_x_wg_i32(p, a); \
 \
    work_group_barrier(CLK_LOCAL_MEM_FENCE); \
    if ((i != 0) & (l == 0)) \
        update_##SUF(p, a); \
    work_group_barrier(CLK_LOCAL_MEM_FENCE); \
    a = __llvm_ld_atomic_a3_x_wg_i32(p); \
    work_group_barrier(CLK_LOCAL_MEM_FENCE); \
 \
    return a; \
}

GEN_AA(all, 1U)
GEN_AA(any, 0U);

