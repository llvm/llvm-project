/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#pragma once
#include "asan_util.h"

//offset from llvm/compiler-rt/lib/asan/asan_mapping.h
static const u64 kh_Linux64bit_ShadowOffset =
    0x7FFFFFFF & (~0xFFFULL << ASAN_SHADOW);

#define MEM_TO_SHADOW(mem_addr) (((mem_addr) >> ASAN_SHADOW) + kh_Linux64bit_ShadowOffset)

// Addresses are atleast SHADOW_GRANULARITY aligned.
// True, when given byte is accessible false otherwise.
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

NO_SANITIZE_ADDR
uptr
__asan_region_is_poisoned(uptr beg, uptr size);
