
#include "irif.h"
#include "device_amd_hsa.h"
#include "ockl.h"

__attribute__((always_inline, const)) bool
OCKL_MANGLE_T(is_local,addr)(const void *a)
{
    __constant amd_queue_t *q = (__constant amd_queue_t *)__llvm_amdgcn_queue_ptr();
    uint u = (uint)((ulong)a >> 32);
    return u == q->group_segment_aperture_base_hi;
}

__attribute__((always_inline, const)) bool
OCKL_MANGLE_T(is_private,addr)(const void *a)
{
    __constant amd_queue_t *q = (__constant amd_queue_t *)__llvm_amdgcn_queue_ptr();
    uint u = (uint)((ulong)a >> 32);
    return u == q->private_segment_aperture_base_hi;
}

__attribute__((always_inline, const)) __global void *
OCKL_MANGLE_T(to,global)(void *a)
{
    __global void *ga = (__global void *)((ulong)a);
    return OCKL_MANGLE_T(is_local,addr)(a) | OCKL_MANGLE_T(is_private,addr)(a) ?  (__global void *)0UL : ga;
}

__attribute__((always_inline, const)) __local void *
OCKL_MANGLE_T(to,local)(void *a)
{
    uint u = (uint)((ulong)a);
    return OCKL_MANGLE_T(is_local,addr)(a) ? (__local void *)u : (__local void *)0;
}

__attribute__((always_inline, const)) __private void *
OCKL_MANGLE_T(to,private)(void *a)
{
    uint u = (uint)((ulong)a);
    return OCKL_MANGLE_T(is_private,addr)(a) ? (__private void *)u : (__private void *)0;
}

