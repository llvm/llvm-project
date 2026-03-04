/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "asan_util.h"
#include "shadow_mapping.h"

static const __constant u8 kAsanArrayCookieMagic = (u8)0xac;
static const __constant u8 kAsanHeapFreeMagic = (u8)0xfd;

USED NO_SANITIZE_ADDR
void
__asan_poison_cxx_array_cookie(uptr a) {
    __global u8 *sa = (__global u8 *)MEM_TO_SHADOW(a);
    *sa = kAsanArrayCookieMagic;
}

USED NO_INLINE NO_SANITIZE_ADDR
uptr
__asan_load_cxx_array_cookie(uptr a) {
    uptr pc = GET_CALLER_PC();
    __global u8 *sa = (__global u8 *)MEM_TO_SHADOW(a);
    u8 sv = *sa;
    if (sv == kAsanArrayCookieMagic)
        return *(__global uptr *)a;
    if (sv == kAsanHeapFreeMagic) {
        REPORT_IMPL(pc, a, 0, 1, false);
        return 0;
    }
    return *(__global uptr *)a;
}

