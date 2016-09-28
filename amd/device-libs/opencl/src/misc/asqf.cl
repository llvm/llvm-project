
#include "ockl.h"

__attribute__((overloadable, always_inline, const)) cl_mem_fence_flags
get_fence(void *a)
{
    return OCKL_MANGLE_T(is_local,addr)(a) ? CLK_LOCAL_MEM_FENCE : CLK_GLOBAL_MEM_FENCE;
}

__attribute__((overloadable, always_inline, const)) cl_mem_fence_flags
get_fence(const void *a)
{
    return OCKL_MANGLE_T(is_local,addr)(a) ? CLK_LOCAL_MEM_FENCE : CLK_GLOBAL_MEM_FENCE;
}

__attribute__((always_inline, const)) __global void *
__to_global(void *a)
{
    return OCKL_MANGLE_T(to,global)(a);
}

__attribute__((always_inline, const)) __local void *
__to_local(void *a)
{
    return OCKL_MANGLE_T(to,local)(a);
}

__attribute__((always_inline, const)) __private void *
__to_private(void *a)
{
    return OCKL_MANGLE_T(to,private)(a);
}

