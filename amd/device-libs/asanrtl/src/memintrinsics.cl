/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "asan_util.h"
#include "shadow_mapping.h"

OPT_NONE
NO_SANITIZE_ADDR
static void
check_memory_range_accessible(void* dest, const void* src,
                    uptr size, uptr pc) {
    if (size == 0)
      return;
    uptr invalid_addr = 0;
    uptr src_addr = (uptr)src;
    invalid_addr = __asan_region_is_poisoned(src_addr, size);
    if (invalid_addr) {
      REPORT_IMPL(pc, invalid_addr, false, size, false)
    }
    uptr dest_addr = (uptr)dest;
    invalid_addr = __asan_region_is_poisoned(dest_addr, size);
    if (invalid_addr) {
      REPORT_IMPL(pc, invalid_addr, true, size, false)
    }
}

OPT_NONE
NO_SANITIZE_ADDR
void*
__asan_memcpy(void* to, const void* from, uptr size) {
    uptr pc = GET_CALLER_PC();
    check_memory_range_accessible(to, from, size, pc);
    return __builtin_memcpy(to, from, size);
}

OPT_NONE
NO_SANITIZE_ADDR
void*
__asan_memmove(void* to, const void* from, uptr size) {
    uptr pc = GET_CALLER_PC();
    check_memory_range_accessible(to, from, size, pc);
    return __builtin_memmove(to, from, size);
}

OPT_NONE
NO_SANITIZE_ADDR
void*
__asan_memset(void* s, int c, uptr n) {
    uptr pc = GET_CALLER_PC();
    uptr src_addr = (uptr)s;
    uptr invalid_addr = 0;
    invalid_addr = __asan_region_is_poisoned(src_addr, n);
    if (invalid_addr) {
      REPORT_IMPL(pc, invalid_addr, true, n, false)
    }
    return __builtin_memset(s, c, n);
}
