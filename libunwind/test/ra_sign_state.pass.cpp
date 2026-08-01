// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// clang-format off

// REQUIRES: target={{aarch64.*}}
// UNSUPPORTED: target={{.*-windows.*}}

// The libSystem unwinder does not correctly read UNW_AARCH64_RA_SIGN_STATE, at
// least through OS version 27.0
// XFAIL: stdlib=apple-libc++ && target={{.*}}-apple-{{.*}}{{(11|12|13|14|15|26)(\.\d+)?}}
// XFAIL: stdlib=apple-libc++ && target={{.*}}-apple-{{.*}}27.0

// clang-format on

#undef NDEBUG
#include "../src/config.h"
#include "support/func_bounds.h"

#include <assert.h>
#include <inttypes.h>
#include <libunwind.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <unistd.h>
#include <unwind.h>

#if defined(__APPLE__)
#include <sys/sysctl.h>
#endif
#if defined(_LIBUNWIND_HAVE_GETAUXVAL) || defined(_LIBUNWIND_HAVE_ELF_AUX_INFO)
#include <sys/auxv.h>
#endif

// Note: This test requires FEAT_PAuth (and is setup to pass on other targets).

#if defined(__APPLE__)
static bool checkHasPAuth() {
  int has_pauth = 0;
  size_t size = sizeof(has_pauth);
  if (sysctlbyname("hw.optional.arm.FEAT_PAuth", &has_pauth, &size, NULL, 0))
    return false;
  return has_pauth != 0;
}
#elif defined(_LIBUNWIND_HAVE_GETAUXVAL)
static bool checkHasPAuth() {
  constexpr unsigned long hwcap_paca = (1UL << 30);
  unsigned long hwcap = getauxval(AT_HWCAP);
  return (hwcap & hwcap_paca) != 0;
}
#elif defined(_LIBUNWIND_HAVE_ELF_AUX_INFO)
static bool checkHasPAuth() {
  constexpr unsigned long hwcap_paca = (1UL << 30);
  unsigned long hwcap = 0;
  elf_aux_info(AT_HWCAP, &hwcap, sizeof(hwcap));
  return (hwcap & hwcap_paca) != 0;
}
#else
static bool checkHasPAuth() {
  // TODO: Support other platforms.
  return false;
}
#endif

FUNC_BOUNDS_DECL(main_func);

static _Unwind_Reason_Code frame_handler(struct _Unwind_Context *ctx,
                                         void *arg) {
  fprintf(stderr, "frame_handler\n");
  uint64_t ra_sign_state =
      (uint64_t)_Unwind_GetGR(ctx, UNW_AARCH64_RA_SIGN_STATE);

  fprintf(stderr, "ra_sign_state = 0x%" PRIx64 "\n", ra_sign_state);

  uintptr_t ip = _Unwind_GetIP(ctx);

  fprintf(stderr, "UNW_AARCH64_RA_SIGN_STATE @ 0x%" PRIxPTR " = %" PRIu64 "\n",
          ip, ra_sign_state);

  if (ip >= (uintptr_t)FUNC_START(main_func) &&
      ip < (uintptr_t)FUNC_END(main_func)) {

    // Collect the RA from the callee that will return to main.
    *(uint64_t *)arg = ra_sign_state;

    // Unwind until main is reached, above frames depend on the platform and
    // architecture.
    return _URC_END_OF_STACK;
  }

#if defined(_LIBUNWIND_TARGET_AARCH64_AUTHENTICATED_UNWINDING)
  assert(ra_sign_state == 1 || ra_sign_state == 2);
#endif

  return _URC_NO_REASON;
}

__attribute__((noinline)) extern "C" uintptr_t get_main_ra_sign_state() {
  uint64_t sign_state = -1;
  _Unwind_Backtrace(frame_handler, &sign_state);
  fprintf(stderr, "get_main_ra_sign_state: sign_state = 0x%" PRIx64 "\n",
          sign_state);
  assert((sign_state & 0x3) == sign_state);
  return sign_state;
}

__attribute__((noinline)) static uint64_t check_vanilla() {
  return get_main_ra_sign_state();
}

__attribute__((naked, target("pauth"))) static uint64_t check_negate() {
  // clang-format off
  asm(
#if !defined(__APPLE__)
      ".cfi_b_key_frame\n"
#endif
      ".cfi_negate_ra_state\n"
      "pacibsp\n"

      "stp x29, x30, [sp, #-16]!\n"
      ".cfi_def_cfa_offset 16\n"
      ".cfi_offset x29, -16\n"
      ".cfi_offset x30, -8\n"

      "bl " SYMBOL_NAME(get_main_ra_sign_state) "\n"

      "ldp x29, x30, [sp], #16\n"
      ".cfi_def_cfa_offset 0\n"
      ".cfi_restore x29\n"
      ".cfi_restore x30\n"

      ".cfi_negate_ra_state\n"
      "retab");
  // clang-format on
}

#if defined(HAVE_CFI_SET_RA_STATE)
__attribute__((naked, target("pauth"))) uint64_t check_set() {
  // clang-format off
  asm(
#if !defined(__APPLE__)
      ".cfi_b_key_frame\n"
#endif
      ".cfi_set_ra_state 1, 0\n"
      "pacibsp\n"

      "stp x29, x30, [sp, #-16]!\n"
      ".cfi_def_cfa_offset 16\n"
      ".cfi_offset x29, -16\n"
      ".cfi_offset x30, -8\n"

      "bl " SYMBOL_NAME(get_main_ra_sign_state) "\n"

      "ldp x29, x30, [sp], #16\n"
      ".cfi_def_cfa_offset 0\n"
      ".cfi_restore x29\n"
      ".cfi_restore x30\n"

      ".cfi_set_ra_state 0, -20\n"
      "retab");
  // clang-format on
}
#endif

FUNC_ATTR(main_func) int main(int, char **) {
  uint64_t ret;

  ret = check_vanilla();
  fprintf(stderr, "check_vanilla: ret = 0x%" PRIx64 "\n", ret);
#if defined(_LIBUNWIND_TARGET_AARCH64_AUTHENTICATED_UNWINDING)
  assert(ret == 1 || ret == 2);
#endif

  if (!checkHasPAuth()) {
    fprintf(stderr, "target does not have FEAT_PAuth\n");
    return 0;
  }

  ret = check_negate();
  fprintf(stderr, "check_negate: ret = 0x%" PRIx64 "\n", ret);
  assert(ret == 1);

#if defined(HAVE_CFI_SET_RA_STATE)
  ret = check_set();
  fprintf(stderr, "check_set: ret = 0x%" PRIx64 "\n", ret);
  assert(ret == 1);
#endif

  fprintf(stderr, "success\n");
  return 0;
}
