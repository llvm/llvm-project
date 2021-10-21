/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#pragma once
#include "ockl.h"

typedef ulong uptr;
typedef unsigned char u8;
typedef signed char s8;
typedef unsigned short u16;
typedef short s16;
typedef unsigned long u64;

#define ASAN_SHADOW 3

#define SHADOW_GRANULARITY (1ULL << ASAN_SHADOW)

#define GET_CALLER_PC() (uptr) __builtin_return_address(0)

#define WORKGROUP_ID(dim) __builtin_amdgcn_workgroup_id_##dim()

#define OPT_NONE __attribute__((optnone))

#define NO_SANITIZE_ADDR __attribute__((no_sanitize("address")))

#define REPORT_IMPL(caller_pc, addr, is_write, size, no_abort)                 \
    uptr read = is_write;                                                      \
    if (no_abort)                                                              \
        read |= 0xFFFFFFFF00000000;                                            \
                                                                               \
    __ockl_sanitizer_report(addr, caller_pc, WORKGROUP_ID(x), WORKGROUP_ID(y), \
                            WORKGROUP_ID(z), __ockl_get_local_linear_id(),     \
                            read, size);

NO_SANITIZE_ADDR
static bool
is_aligned_by_granularity(uptr addr)
{
    return (addr & (SHADOW_GRANULARITY - 1)) == 0;
}

// round up size to the nearest multiple of boundary.
NO_SANITIZE_ADDR
static uptr
round_upto(uptr size, uptr boundary)
{
    return (size + boundary - 1) & ~(boundary - 1);
}

// round down size to the nearest multiple of boundary.
NO_SANITIZE_ADDR
static uptr
round_downto(uptr size, uptr boundary)
{
    return size & ~(boundary - 1);
}
