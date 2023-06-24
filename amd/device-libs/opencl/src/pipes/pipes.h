/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "ockl.h"

#pragma OPENCL EXTENSION cl_khr_int64_base_atomics : enable
#pragma OPENCL EXTENSION cl_khr_int64_extended_atomics : enable

extern size_t __amd_wresvn(volatile __global atomic_size_t *pidx, size_t lim, size_t n);

#define DO_PIPE_SIZE(F) \
F(1,uchar) \
F(2,ushort) \
F(4,uint) \
F(8,ulong) \
F(16,ulong2) \
F(32,ulong4) \
F(64,ulong8) \
F(128,ulong16)

struct pipeimp {
    atomic_size_t read_idx;
    atomic_size_t write_idx;
    size_t end_idx;
    uchar pad[128 - 3*sizeof(size_t)];
    uchar packets[1];
};

extern void __memcpy_internal_aligned(void *, const void *, size_t, size_t);

static __attribute__((always_inline)) size_t
reserve(volatile __global atomic_size_t *pi, size_t lim, size_t n)
{
    size_t i = __opencl_atomic_load(pi, memory_order_relaxed, memory_scope_device);

    for (;;) {
        if (i + n > lim)
            return ~(size_t)0;

        if (__opencl_atomic_compare_exchange_strong(pi, &i, i + n, memory_order_relaxed, memory_order_relaxed, memory_scope_device))
            break;
    }

    return i;
}

static inline size_t
wave_reserve_1(volatile __global atomic_size_t *pi, size_t lim)
{
    ulong n = __builtin_popcountl(__builtin_amdgcn_read_exec());
    uint l = __builtin_amdgcn_mbcnt_hi(__builtin_amdgcn_read_exec_hi(),
               __builtin_amdgcn_mbcnt_lo(__builtin_amdgcn_read_exec_lo(), 0u));
    size_t i = 0;

    if (l == 0) {
        i = __opencl_atomic_load(pi, memory_order_relaxed, memory_scope_device);

        for (;;) {
            if (i + n > lim) {
                i = ~(size_t)0;
                break;
            }

            if (__opencl_atomic_compare_exchange_strong(pi, &i, i + n, memory_order_relaxed, memory_order_relaxed, memory_scope_device))
                break;
        }
    }

    __builtin_amdgcn_wave_barrier();

    // Broadcast the result; the ctz tells us which lane has active lane id 0
    uint k = (uint)OCKL_MANGLE_U64(ctz)(__builtin_amdgcn_read_exec());
    i = ((size_t)__builtin_amdgcn_readlane((uint)(i >> 32), k) << 32) |
        (size_t)__builtin_amdgcn_readlane((uint)i, k);

    __builtin_amdgcn_wave_barrier();

    if (i != ~(size_t)0)
        i += l;
    else {
        // The entire group didn't fit, have to handle one by one
        i = reserve(pi, lim, (size_t)1);
    }

    return i;
}

static inline size_t
wrap(size_t i, size_t n)
{
    // Assume end_i < 2^32
    size_t ret;
    if (as_uint2(i).y == 0U) {
        uint j = (uint)i;
        uint m = (uint)n;
        if (j < m)
            ret = i;
        else
            ret = (ulong)(j % m);
    } else
        ret = i % n;
    return ret;
}

