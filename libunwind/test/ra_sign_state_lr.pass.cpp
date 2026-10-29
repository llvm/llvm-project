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
// UNSUPPORTED: target={{.*-apple.*}}

// clang-format on

#undef NDEBUG
#include "../src/config.h"
#include "support/func_bounds.h"

#include <assert.h>
#include <inttypes.h>
#include <libunwind.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <unistd.h>
#include <unwind.h>

#if defined(_LIBUNWIND_HAVE_GETAUXVAL) || defined(_LIBUNWIND_HAVE_ELF_AUX_INFO)
#include <asm/hwcap.h>
#include <sys/auxv.h>
#endif

// Note: This test requires FEAT_PAuth_LR (and is setup to pass on other
// targets).

static bool hasPAuthLR() {
  uint64_t isa_r1, isa_r2;
  asm("mrs %0, ID_AA64ISAR1_EL1" : "=r"(isa_r1));
  asm("mrs %0, ID_AA64ISAR2_EL1" : "=r"(isa_r2));

  uint64_t api = (isa_r1 >> 8) & 0xf;
  uint64_t apa = (isa_r1 >> 4) & 0xf;
  uint64_t apa3 = (isa_r2 >> 12) & 0xf;

  return api == 6 || apa == 6 || apa3 == 6;
}

#if defined(_LIBUNWIND_HAVE_GETAUXVAL)
static bool checkHasPAuthLR() {
  unsigned long hwcap = getauxval(AT_HWCAP);
  if ((hwcap & HWCAP_CPUID) == 0)
    return false;

  return hasPAuthLR();
}
#elif defined(_LIBUNWIND_HAVE_ELF_AUX_INFO)
static bool checkHasPAuthLR() {
  unsigned long hwcap;
  if (elf_aux_info(AT_HWCAP, &hwcap, sizeof(hwcap)) != 0)
    return false;

  if ((hwcap & HWCAP_CPUID) == 0)
    return false;

  return hasPAuthLR();
}
#elif defined(_LIBUNWIND_IS_BAREMETAL)
static bool checkHasPAuthLR() { return hasPAuthLR(); }
#else
static bool checkHasPAuthLR() {
  // TODO: Support other platforms.
  return false;
}
#endif

FUNC_BOUNDS_DECL(main_func);

static _Unwind_Reason_Code frame_handler(struct _Unwind_Context *ctx,
                                         void *arg) {
  fprintf(stderr, "frame_handler\n");
  uintptr_t ip = _Unwind_GetIP(ctx);

  if (ip >= (uintptr_t)FUNC_START(main_func) &&
      ip < (uintptr_t)FUNC_END(main_func)) {
    *(bool *)arg = true;

    // Unwind until main is reached, above frames depend on the platform and
    // architecture.
    return _URC_END_OF_STACK;
  }

  return _URC_NO_REASON;
}

__attribute__((noinline)) extern "C" uintptr_t reaches_main_via_backtrace() {
  bool reached_main = false;
  _Unwind_Backtrace(frame_handler, &reached_main);
  fprintf(stderr, "reached_main = %d\n", reached_main);
  return reached_main;
}

__attribute__((noinline)) static uint64_t check_vanilla() {
  return reaches_main_via_backtrace();
}

#if defined(HAVE_CFI_SET_RA_STATE)
__attribute__((naked, target("pauth-lr"))) uint64_t check_set() {
  // clang-format off
  asm(
      ".cfi_b_key_frame\n"
      ".cfi_non_hint_pauth_frame\n"
      ".Lsign:\n"
      "pacibsppc\n"
      ".cfi_set_ra_state 2, .Lsign\n"

      "stp x29, x30, [sp, #-16]!\n"
      ".cfi_def_cfa_offset 16\n"
      ".cfi_offset x29, -16\n"
      ".cfi_offset x30, -8\n"

      "bl " SYMBOL_NAME(reaches_main_via_backtrace) "\n"

      "ldp x29, x30, [sp], #16\n"
      ".cfi_def_cfa_offset 0\n"
      ".cfi_restore x29\n"
      ".cfi_restore x30\n"

      ".cfi_set_ra_state 0, -20\n"
      "retabsppc .Lsign");
  // clang-format on
}
#endif

FUNC_ATTR(main_func) int main(int, char **) {
  uint64_t ret;

  ret = check_vanilla();
  fprintf(stderr, "check_vanilla: ret = 0x%" PRIx64 "\n", ret);
  assert(ret == 1);

  if (!checkHasPAuthLR()) {
    fprintf(stderr, "target does not have FEAT_PAuth_LR\n");
    return 0;
  }

#if defined(HAVE_CFI_SET_RA_STATE)
  ret = check_set();
  fprintf(stderr, "check_set: ret = 0x%" PRIx64 "\n", ret);
  assert(ret == 1);
#endif

  fprintf(stderr, "success\n");
  return 0;
}
