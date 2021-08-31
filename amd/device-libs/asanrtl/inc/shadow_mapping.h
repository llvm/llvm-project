/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#pragma once
#include "asan_type_decls.h"

#define ASAN_SHADOW 3

#define SHADOW_GRANULARITY (1ULL << ASAN_SHADOW)

//offset from llvm/compiler-rt/lib/asan/asan_mapping.h
static const u64 kh_Linux64bit_ShadowOffset =
    0x7FFFFFFF & (~0xFFFULL << ASAN_SHADOW);

#define MEM_TO_SHADOW(mem_addr) (((mem_addr) >> ASAN_SHADOW) + kh_Linux64bit_ShadowOffset)

#define NO_SANITIZE_ADDR __attribute__((no_sanitize("address")))

//address are atleast SHADOW_GRANULARITY aligned
//true, when given byte is accessible false otherwise
NO_SANITIZE_ADDR
static bool
is_address_poisoned(uptr addr)
{
    uptr shadow_addr = MEM_TO_SHADOW(addr);
    s8 shadow_value = *(__global s8 *)shadow_addr;
    if (shadow_value) {
        //compute index of the given address within 8-byte range
        return (s8)(addr & (SHADOW_GRANULARITY - 1)) >= shadow_value;
    }
    return false;
}

//check all application bytes in [beg,beg+size) range are accessible
NO_SANITIZE_ADDR
static bool
is_region_poisoned(uptr beg, uptr size)
{
    uptr end = beg + size - 1;
    // Fast path - check first and last application bytes
    if (is_address_poisoned(beg) ||
        is_address_poisoned(end))
    return true;

    // check all inner bytes
    for (uptr addr = beg+1; addr < end; addr++){
       if (is_address_poisoned(addr))
        return true;
    }
    return false;
}
