/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "asan_util.h"
#include "shadow_mapping.h"

#define ASAN_REPORT_ERROR(type, size, is_write)                                  \
USED NO_INLINE NO_SANITIZE_ADDR                                                  \
void __asan_report_ ## type ## size(uptr addr) {                                 \
    REPORT_IMPL(GET_CALLER_PC(), addr, is_write, size, false)                    \
}                                                                                \
USED NO_INLINE NO_SANITIZE_ADDR                                                  \
void __asan_report_ ## type ## size ## _noabort(uptr addr) {                     \
    REPORT_IMPL(GET_CALLER_PC(), addr, is_write, size, true)                     \
}                                                                                \

ASAN_REPORT_ERROR(load, 1, 0)
ASAN_REPORT_ERROR(load, 2, 0)
ASAN_REPORT_ERROR(load, 4, 0)
ASAN_REPORT_ERROR(load, 8, 0)
ASAN_REPORT_ERROR(load, 16,0)

ASAN_REPORT_ERROR(store, 1, 1)
ASAN_REPORT_ERROR(store, 2, 1)
ASAN_REPORT_ERROR(store, 4, 1)
ASAN_REPORT_ERROR(store, 8, 1)
ASAN_REPORT_ERROR(store, 16,1)

#define ASAN_REPORT_ERROR_N(type, is_write)                        \
USED NO_INLINE NO_SANITIZE_ADDR                                    \
void __asan_report_ ## type ## _n(uptr addr, uptr size) {          \
    REPORT_IMPL(GET_CALLER_PC(), addr, is_write, size, false)      \
}                                                                  \
USED NO_INLINE NO_SANITIZE_ADDR                                    \
void __asan_report_ ## type ## _n_noabort(uptr addr, uptr size) {  \
    REPORT_IMPL(GET_CALLER_PC(), addr, is_write, size, true)       \
}                                                                  \

ASAN_REPORT_ERROR_N(store,1)
ASAN_REPORT_ERROR_N(load,0)

NO_SANITIZE_ADDR
static bool
is_invalid_access(uptr addr, uptr size)
{
    uptr shadow_addr = MEM_TO_SHADOW(addr);
    if (size <= SHADOW_GRANULARITY) {
      s8 shadow_value = *(__global s8*) shadow_addr;
      return shadow_value != 0 && ((s8)((addr & (SHADOW_GRANULARITY-1)) + size - 1) >= shadow_value);
    }
    else {
      s16 shadow_value = *(__global s16*) shadow_addr;
      return shadow_value != 0;
    }
}

#define ASAN_ERROR(type, size, is_write)                     \
USED NO_INLINE NO_SANITIZE_ADDR                              \
void __asan_ ## type ## size(uptr addr) {                    \
    uptr caller_pc = GET_CALLER_PC();                        \
    if (is_invalid_access(addr, size)) {                     \
        REPORT_IMPL(caller_pc, addr, is_write, size, false)  \
    }                                                        \
}                                                            \
USED NO_INLINE NO_SANITIZE_ADDR                              \
void __asan_ ## type ## size ## _noabort(uptr addr) {        \
    uptr caller_pc = GET_CALLER_PC();                        \
    if (is_invalid_access(addr, size)) {                     \
        REPORT_IMPL(caller_pc, addr, is_write, size, true)   \
    }                                                        \
}                                                            \

ASAN_ERROR(load, 1, 0)
ASAN_ERROR(load, 2, 0)
ASAN_ERROR(load, 4, 0)
ASAN_ERROR(load, 8, 0)
ASAN_ERROR(load, 16,0)

ASAN_ERROR(store, 1, 1)
ASAN_ERROR(store, 2, 1)
ASAN_ERROR(store, 4, 1)
ASAN_ERROR(store, 8, 1)
ASAN_ERROR(store, 16,1)

#define ASAN_ERROR_N(type, is_write)                         \
USED NO_INLINE NO_SANITIZE_ADDR                              \
void __asan_ ## type ## _n(uptr addr, uptr size) {           \
    uptr caller_pc = GET_CALLER_PC();                        \
    if (__asan_region_is_poisoned(addr, size)) {             \
        REPORT_IMPL(caller_pc, addr, is_write, size, false)  \
    }                                                        \
}                                                            \
USED NO_INLINE NO_SANITIZE_ADDR                              \
void __asan_ ## type ## _n_noabort(uptr addr, uptr size) {   \
    uptr caller_pc = GET_CALLER_PC();                        \
    if (__asan_region_is_poisoned(addr, size)) {             \
        REPORT_IMPL(caller_pc, addr, is_write, size, true)   \
    }                                                        \
}                                                            \

ASAN_ERROR_N(store, 1)
ASAN_ERROR_N(load, 0)
