
/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "irif.h"
#include "ockl.h"
#include "oclc.h"

__attribute__((target("s-memrealtime"))) void
OCKL_MANGLE_T(rtcwait,u32)(uint ticks)
{
    ulong now = __builtin_amdgcn_s_memrealtime();
    ulong end = now + __builtin_amdgcn_readfirstlane(ticks);

    if (__oclc_ISA_version >= 9000) {
        while (end > now + 1625) {
            __builtin_amdgcn_s_sleep(127);
            now = __builtin_amdgcn_s_memrealtime();
        }

        while (end > now + 806) {
            __builtin_amdgcn_s_sleep(63);
            now = __builtin_amdgcn_s_memrealtime();
        }

        while (end > now + 396) {
            __builtin_amdgcn_s_sleep(31);
            now = __builtin_amdgcn_s_memrealtime();
        }
    }

    while (end > now + 192) {
        __builtin_amdgcn_s_sleep(15);
        now = __builtin_amdgcn_s_memrealtime();
    }

    while (end > now + 89) {
        __builtin_amdgcn_s_sleep(7);
        now = __builtin_amdgcn_s_memrealtime();
    }

    while (end > now + 38) {
        __builtin_amdgcn_s_sleep(3);
        now = __builtin_amdgcn_s_memrealtime();
    }

    while (end > now) {
        __builtin_amdgcn_s_sleep(1);
        now = __builtin_amdgcn_s_memrealtime();
    }
}

