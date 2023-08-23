
/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "oclc.h"
#include "ockl.h"

static uint
bpermute_u32(uint l, uint v)
{
    return __builtin_amdgcn_ds_bpermute(l << 2, v);
}

uint
OCKL_MANGLE_U32(alisa)(uint n)
{
    uint l = __ockl_lane_u32();
    uint ret = n;

    if (__oclc_wavefrontsize64) {
        const ulong nolsb = ~1UL;

        // Step 1
        ulong smask = __builtin_amdgcn_read_exec() & ~((0x2UL << l) - 0x1UL);
        int slid = (int)__ockl_ctz_u64(smask);
        uint t = bpermute_u32(slid, n);
        ret += slid < 64 ? t : 0;
        smask ^= (0x1UL << slid) & nolsb;

        // Step 2
        slid = (int)__ockl_ctz_u64(smask);
        t = bpermute_u32(slid, ret);
        ret += slid < 64 ? t : 0;

        smask ^= (0x1UL << slid) & nolsb;
        slid = (int)__ockl_ctz_u64(smask);
        smask ^= (0x1UL << slid) & nolsb;

        // Step 3
        slid = __ockl_ctz_u64(smask);
        t = bpermute_u32(slid, ret);
        ret += slid < 64 ? t : 0;

        smask ^= (0x1UL << slid) & nolsb;
        slid = __ockl_ctz_u64(smask);
        smask ^= (0x1UL << slid) & nolsb;
        slid = __ockl_ctz_u64(smask);
        smask ^= (0x1UL << slid) & nolsb;
        slid = __ockl_ctz_u64(smask);
        smask ^= (0x1UL << slid) & nolsb;

        // Step 4
        slid = __ockl_ctz_u64(smask);
        t = bpermute_u32(slid, ret);
        ret += slid < 64 ? t : 0;

        smask ^= (0x1UL << slid) & nolsb;
        slid = __ockl_ctz_u64(smask);
        smask ^= (0x1UL << slid) & nolsb;
        slid = __ockl_ctz_u64(smask);
        smask ^= (0x1UL << slid) & nolsb;
        slid = __ockl_ctz_u64(smask);
        smask ^= (0x1UL << slid) & nolsb;
        slid = __ockl_ctz_u64(smask);
        smask ^= (0x1UL << slid) & nolsb;
        slid = __ockl_ctz_u64(smask);
        smask ^= (0x1UL << slid) & nolsb;
        slid = __ockl_ctz_u64(smask);
        smask ^= (0x1UL << slid) & nolsb;
        slid = __ockl_ctz_u64(smask);
        smask ^= (0x1UL << slid) & nolsb;

        // Step 5
        slid = __ockl_ctz_u64(smask);
        t = bpermute_u32(slid, ret);
        ret += slid < 64 ? t : 0;

        smask ^= (0x1UL << slid) & nolsb;
        slid = __ockl_ctz_u64(smask);
        smask ^= (0x1UL << slid) & nolsb;
        slid = __ockl_ctz_u64(smask);
        smask ^= (0x1UL << slid) & nolsb;
        slid = __ockl_ctz_u64(smask);
        smask ^= (0x1UL << slid) & nolsb;
        slid = __ockl_ctz_u64(smask);
        smask ^= (0x1UL << slid) & nolsb;
        slid = __ockl_ctz_u64(smask);
        smask ^= (0x1UL << slid) & nolsb;
        slid = __ockl_ctz_u64(smask);
        smask ^= (0x1UL << slid) & nolsb;
        slid = __ockl_ctz_u64(smask);
        smask ^= (0x1UL << slid) & nolsb;
        slid = __ockl_ctz_u64(smask);
        smask ^= (0x1UL << slid) & nolsb;
        slid = __ockl_ctz_u64(smask);
        smask ^= (0x1UL << slid) & nolsb;
        slid = __ockl_ctz_u64(smask);
        smask ^= (0x1UL << slid) & nolsb;
        slid = __ockl_ctz_u64(smask);
        smask ^= (0x1UL << slid) & nolsb;
        slid = __ockl_ctz_u64(smask);
        smask ^= (0x1UL << slid) & nolsb;
        slid = __ockl_ctz_u64(smask);
        smask ^= (0x1UL << slid) & nolsb;
        slid = __ockl_ctz_u64(smask);
        smask ^= (0x1UL << slid) & nolsb;
        slid = __ockl_ctz_u64(smask);
        smask ^= (0x1UL << slid) & nolsb;

        // Step 6
        slid = __ockl_ctz_u64(smask);
        t = bpermute_u32(slid, ret);
        ret += slid < 64 ? t : 0;
    } else {
        const uint nolsb = ~1U;

        // Step 1
        uint smask = __builtin_amdgcn_read_exec_lo() & ~((0x2U << l) - 0x1U);
        int slid = (int)__ockl_ctz_u32(smask);
        uint t = bpermute_u32(slid, n);
        ret += slid < 32 ? t : 0;
        smask ^= (0x1U << slid) & nolsb;

        // Step 2
        slid = (int)__ockl_ctz_u32(smask);
        t = bpermute_u32(slid, ret);
        ret += slid < 32 ? t : 0;

        smask ^= (0x1U << slid) & nolsb;
        slid = (int)__ockl_ctz_u32(smask);
        smask ^= (0x1U << slid) & nolsb;

        // Step 3
        slid = __ockl_ctz_u32(smask);
        t = bpermute_u32(slid, ret);
        ret += slid < 32 ? t : 0;

        smask ^= (0x1U << slid) & nolsb;
        slid = __ockl_ctz_u32(smask);
        smask ^= (0x1U << slid) & nolsb;
        slid = __ockl_ctz_u32(smask);
        smask ^= (0x1U << slid) & nolsb;
        slid = __ockl_ctz_u32(smask);
        smask ^= (0x1U << slid) & nolsb;

        // Step 4
        slid = __ockl_ctz_u32(smask);
        t = bpermute_u32(slid, ret);
        ret += slid < 32 ? t : 0;

        smask ^= (0x1U << slid) & nolsb;
        slid = __ockl_ctz_u32(smask);
        smask ^= (0x1U << slid) & nolsb;
        slid = __ockl_ctz_u32(smask);
        smask ^= (0x1U << slid) & nolsb;
        slid = __ockl_ctz_u32(smask);
        smask ^= (0x1U << slid) & nolsb;
        slid = __ockl_ctz_u32(smask);
        smask ^= (0x1U << slid) & nolsb;
        slid = __ockl_ctz_u32(smask);
        smask ^= (0x1U << slid) & nolsb;
        slid = __ockl_ctz_u32(smask);
        smask ^= (0x1U << slid) & nolsb;
        slid = __ockl_ctz_u32(smask);
        smask ^= (0x1U << slid) & nolsb;

        // Step 5
        slid = __ockl_ctz_u32(smask);
        t = bpermute_u32(slid, ret);
        ret += slid < 32 ? t : 0;
    }

    return ret;
}
