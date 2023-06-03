/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "oclc.h"
#include "ockl.h"

__attribute__((target("s-memrealtime"))) static ulong
mem_realtime(void)
{
    return __builtin_amdgcn_s_memrealtime();
}

__attribute__((target("gfx11-insts"))) static ulong
msg_realtime(void)
{
    return __builtin_amdgcn_s_sendmsg_rtnl(0x83);
}

ulong
OCKL_MANGLE_U64(cyclectr)(void)
{
    return __builtin_readcyclecounter();
}

ulong
OCKL_MANGLE_U64(steadyctr)(void)
{
    if (__oclc_ISA_version >= 11000) {
        return msg_realtime();
    } else {
        return mem_realtime();
    }
}

