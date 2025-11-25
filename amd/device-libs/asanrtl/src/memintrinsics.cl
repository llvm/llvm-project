/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "asan_util.h"
#include "shadow_mapping.h"

NO_SANITIZE_ADDR
static void
check_memory_range_accessible(const void* dst, const void* src, uptr size, uptr pc)
{
    if (size == 0)
      return;

    if (!__ockl_is_private_addr(src) && !__ockl_is_local_addr(src)) {
        uptr invalid_addr = __asan_region_is_poisoned((uptr)src, size);
        if (invalid_addr) {
          REPORT_IMPL(pc, invalid_addr, false, size, false)
        }
    }

    if (!__ockl_is_private_addr(dst) && !__ockl_is_local_addr(dst)) {
        uptr invalid_addr = __asan_region_is_poisoned((uptr)dst, size);
        if (invalid_addr) {
          REPORT_IMPL(pc, invalid_addr, true, size, false)
        }
    }
}

USED
NO_INLINE
NO_SANITIZE_ADDR
void*
__asan_memcpy(void* to, const void* from, uptr size)
{
    uptr pc = GET_CALLER_PC();
    check_memory_range_accessible(to, from, size, pc);
    return __builtin_memcpy(to, from, size);
}

USED
NO_INLINE
NO_SANITIZE_ADDR
void*
__asan_memmove(void* to, const void* from, uptr size)
{
    uptr pc = GET_CALLER_PC();
    check_memory_range_accessible(to, from, size, pc);
    return __builtin_memmove(to, from, size);
}

USED
NO_INLINE
NO_SANITIZE_ADDR
void*
__asan_memset(void* s, int c, uptr n)
{
    uptr pc = GET_CALLER_PC();

    if (!__ockl_is_private_addr(s) && !__ockl_is_local_addr(s)) {
        uptr invalid_addr = __asan_region_is_poisoned((uptr)s, n);
        if (invalid_addr) {
          REPORT_IMPL(pc, invalid_addr, true, n, false)
        }
    }

    return __builtin_memset(s, c, n);
}

