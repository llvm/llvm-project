/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "ockl.h"
#include "asan_type_decls.h"
#include "shadow_mapping.h"

#define GET_CALLER_PC() \
    (uptr) __builtin_return_address(0)

#define WORKGROUP_ID(dim) \
    __builtin_amdgcn_workgroup_id_ ##dim()

#define OPT_NONE __attribute__((optnone))

#define REPORT_IMPL(caller_pc, addr, is_write, size, no_abort)                          \
    uptr read      = is_write;                                                          \
    if (no_abort)                                                                       \
        read       |= 0xFFFFFFFF00000000;                                               \
                                                                                        \
    __ockl_sanitizer_report(addr, caller_pc, WORKGROUP_ID(x), WORKGROUP_ID(y),          \
                          WORKGROUP_ID(z), __ockl_get_local_linear_id(), read, size);   \


#define ASAN_REPORT_ERROR(type, size, is_write)                                  \
OPT_NONE                                                                         \
void __asan_report_ ## type ## size(uptr addr) {                                 \
    REPORT_IMPL(GET_CALLER_PC(), addr, is_write, size, false)                    \
}                                                                                \
OPT_NONE                                                                         \
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
OPT_NONE                                                           \
void __asan_report_ ## type ## _n(uptr addr, uptr size) {          \
    REPORT_IMPL(GET_CALLER_PC(), addr, is_write, size, false)      \
}                                                                  \
OPT_NONE                                                           \
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
OPT_NONE NO_SANITIZE_ADDR                                    \
void __asan_ ## type ## size(uptr addr) {                    \
    uptr caller_pc = GET_CALLER_PC();                        \
    if (is_invalid_access(addr, size)) {                     \
        REPORT_IMPL(caller_pc, addr, is_write, size, false)  \
    }                                                        \
}                                                            \
OPT_NONE NO_SANITIZE_ADDR                                    \
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
OPT_NONE NO_SANITIZE_ADDR                                    \
void __asan_ ## type ## _n(uptr addr, uptr size) {           \
    uptr caller_pc = GET_CALLER_PC();                        \
    if (is_region_poisoned(addr, size)) {                    \
        REPORT_IMPL(caller_pc, addr, is_write, size, false)  \
    }                                                        \
}                                                            \
OPT_NONE NO_SANITIZE_ADDR                                    \
void __asan_ ## type ## _n_noabort(uptr addr, uptr size) {   \
    uptr caller_pc = GET_CALLER_PC();                        \
    if (is_region_poisoned(addr, size)) {                    \
        REPORT_IMPL(caller_pc, addr, is_write, size, true)   \
    }                                                        \
}                                                            \

ASAN_ERROR_N(store, 1)
ASAN_ERROR_N(load, 0)
