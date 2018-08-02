/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "pipes.h"

size_t
__amd_wresvn(volatile __global atomic_size_t *pidx, size_t lim, size_t n)
{
    uint alc = (size_t)(__builtin_popcount(__builtin_amdgcn_read_exec_lo()) +
                        __builtin_popcount(__builtin_amdgcn_read_exec_hi()));
    uint l = __llvm_amdgcn_mbcnt_hi(-1, __llvm_amdgcn_mbcnt_lo(-1, 0u));
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
        // Inclusive add scan with not all lanes active
        const ulong nomsb = 0x7fffffffffffffffUL;

        // Step 1
        ulong smask = __builtin_amdgcn_read_exec() & ((0x1UL << l) - 0x1UL);
        int slid = 63 - (int)clz(smask);
        uint t = __builtin_amdgcn_ds_bpermute(slid << 2, n);
        uint sum = n + (slid < 0 ? 0 : t);
        smask ^= (0x1UL << slid) & nomsb;

        // Step 2
        slid = 63 - (int)clz(smask);
        t = __builtin_amdgcn_ds_bpermute(slid << 2, sum);
        sum += slid < 0 ? 0 : t;

        smask ^= (0x1UL << slid) & nomsb;
        slid = 63 - (int)clz(smask);
        smask ^= (0x1UL << slid) & nomsb;

        // Step 3
        slid = 63 - (int)clz(smask);
        t = __builtin_amdgcn_ds_bpermute(slid << 2, sum);
        sum += slid < 0 ? 0 : t;

        smask ^= (0x1UL << slid) & nomsb;
        slid = 63 - (int)clz(smask);
        smask ^= (0x1UL << slid) & nomsb;
        slid = 63 - (int)clz(smask);
        smask ^= (0x1UL << slid) & nomsb;
        slid = 63 - (int)clz(smask);
        smask ^= (0x1UL << slid) & nomsb;

        // Step 4
        slid = 63 - (int)clz(smask);
        t = __builtin_amdgcn_ds_bpermute(slid << 2, sum);
        sum += slid < 0 ? 0 : t;

        smask ^= (0x1UL << slid) & nomsb;
        slid = 63 - (int)clz(smask);
        smask ^= (0x1UL << slid) & nomsb;
        slid = 63 - (int)clz(smask);
        smask ^= (0x1UL << slid) & nomsb;
        slid = 63 - (int)clz(smask);
        smask ^= (0x1UL << slid) & nomsb;
        slid = 63 - (int)clz(smask);
        smask ^= (0x1UL << slid) & nomsb;
        slid = 63 - (int)clz(smask);
        smask ^= (0x1UL << slid) & nomsb;
        slid = 63 - (int)clz(smask);
        smask ^= (0x1UL << slid) & nomsb;
        slid = 63 - (int)clz(smask);
        smask ^= (0x1UL << slid) & nomsb;

        // Step 5
        slid = 63 - (int)clz(smask);
        t = __builtin_amdgcn_ds_bpermute(slid << 2, sum);
        sum += slid < 0 ? 0 : t;

        smask ^= (0x1UL << slid) & nomsb;
        slid = 63 - (int)clz(smask);
        smask ^= (0x1UL << slid) & nomsb;
        slid = 63 - (int)clz(smask);
        smask ^= (0x1UL << slid) & nomsb;
        slid = 63 - (int)clz(smask);
        smask ^= (0x1UL << slid) & nomsb;
        slid = 63 - (int)clz(smask);
        smask ^= (0x1UL << slid) & nomsb;
        slid = 63 - (int)clz(smask);
        smask ^= (0x1UL << slid) & nomsb;
        slid = 63 - (int)clz(smask);
        smask ^= (0x1UL << slid) & nomsb;
        slid = 63 - (int)clz(smask);
        smask ^= (0x1UL << slid) & nomsb;
        slid = 63 - (int)clz(smask);
        smask ^= (0x1UL << slid) & nomsb;
        slid = 63 - (int)clz(smask);
        smask ^= (0x1UL << slid) & nomsb;
        slid = 63 - (int)clz(smask);
        smask ^= (0x1UL << slid) & nomsb;
        slid = 63 - (int)clz(smask);
        smask ^= (0x1UL << slid) & nomsb;
        slid = 63 - (int)clz(smask);
        smask ^= (0x1UL << slid) & nomsb;
        slid = 63 - (int)clz(smask);
        smask ^= (0x1UL << slid) & nomsb;
        slid = 63 - (int)clz(smask);
        smask ^= (0x1UL << slid) & nomsb;
        slid = 63 - (int)clz(smask);
        smask ^= (0x1UL << slid) & nomsb;

        // Step 6
        slid = 63 - (int)clz(smask);
        t = __builtin_amdgcn_ds_bpermute(slid << 2, sum);
        sum += slid < 0 ? 0 : t;
        __builtin_amdgcn_wave_barrier();

        size_t idx = 0;
        if (l == 63 - (int)clz(__builtin_amdgcn_read_exec())) {
            idx = reserve(pidx, lim, (size_t)sum);
        }
        __builtin_amdgcn_wave_barrier();

        // Broadcast
        uint k = 63u - (uint)clz(__builtin_amdgcn_read_exec());
        idx = ((size_t)__builtin_amdgcn_readlane((uint)(idx >> 32), k) << 32) |
              (size_t)__builtin_amdgcn_readlane((uint)idx, k);
        __builtin_amdgcn_wave_barrier();

        rid = idx + (size_t)(sum - (uint)n);
        rid = idx != ~(size_t)0 ? rid : idx;
    }

    if (rid == ~(size_t)0) {
        // Try again one at a time
        rid = reserve(pidx, lim, n);
    }

    return rid;
}

