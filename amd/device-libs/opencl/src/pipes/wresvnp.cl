/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "oclc.h"
#include "ockl.h"
#include "pipes.h"

static uint
active_lane_count(void)
{
    if (__oclc_wavefrontsize64) {
        return __builtin_popcountl(__builtin_amdgcn_read_exec());
    } else {
        return __builtin_popcount(__builtin_amdgcn_read_exec_lo());
    }
}

size_t
__amd_wresvn(volatile __global atomic_size_t *pidx, size_t lim, size_t n)
{
    uint alc = active_lane_count();
    uint l = __ockl_lane_u32();
    size_t rid;

    if (__builtin_amdgcn_read_exec() == (1UL << alc) - 1UL) {
        // Handle fully active subgroup
        uint sum = sub_group_scan_inclusive_add((uint)n);
        size_t idx = 0;
        if (l == alc-1) {
            idx = reserve(pidx, lim, (size_t)sum);
        }
        idx = sub_group_broadcast(idx, alc-1);
        rid = idx + (size_t)(sum - (uint)n);
        rid = idx != ~(size_t)0 ? rid : idx;
    } else {
        uint sum = __ockl_alisa_u32((uint)n);
        uint al = __ockl_activelane_u32();

        size_t idx = 0;
        if (al == 0) {
            idx = reserve(pidx, lim, (size_t)sum);
        }
        __builtin_amdgcn_wave_barrier();
        idx = ((size_t)__builtin_amdgcn_readfirstlane((uint)(idx >> 32)) << 32) |
              (size_t)__builtin_amdgcn_readfirstlane((uint)idx);

        rid = idx + (size_t)(sum - (uint)n);
        rid = idx != ~(size_t)0 ? rid : idx;
    }

    if (rid == ~(size_t)0) {
        // Try again one at a time
        rid = reserve(pidx, lim, n);
    }

    return rid;
}

