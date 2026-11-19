// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Verify that the AArch64 `__aarch64_*_preservemost` outline-atomic entry
// points perform the expected atomic updates while preserving caller-saved
// registers guaranteed by `aarch64_outline_atomics_preservemost` (everything
// except x0-x4, x15-x17, and NZCV). Calls are issued via inline assembly so
// this test can run with any AArch64 host compiler regardless of whether it
// supports `-moutline-atomics-preserve-most`.

// REQUIRES: aarch64-target-arch
// REQUIRES: librt_has_outline_atomic_ldadd4_4
// REQUIRES: librt_has_outline_atomic_cas16_4
// REQUIRES: librt_has_aarch64_preservemost
// RUN: %clang_builtins %s %librt -o %t && %run %t

#include <stdint.h>

#if defined(__aarch64__) && !defined(__arm64ec__)

#  define STR1(x) #x
#  define STR(x) STR1(x)
#  define ASM_SYM(name) STR(__USER_LABEL_PREFIX__) #name

// Test a 4-byte fetch-add helper while holding live values in x5 and x14 (the
// preserved GPR boundaries adjacent to clobbered x0-x4 and x15-x17), x8 (the
// AAPCS64 indirect result register), and x13.
static int check_ldadd4_preservemost(void) {
  uint32_t mem = 40;
  register uint64_t x0 __asm__("x0") = 2;
  register uint32_t *x1 __asm__("x1") = &mem;
  register uint64_t x5 __asm__("x5") = 0x0505050550505050ULL;
  register uint64_t x8 __asm__("x8") = 0x0808080880808080ULL;
  register uint64_t x13 __asm__("x13") = 0x0d0d0d0dd0d0d0d0ULL;
  register uint64_t x14 __asm__("x14") = 0x0e0e0e0ee0e0e0e0ULL;
#  ifdef __ARM_FP
  register double d0 __asm__("d0") = 1.25;
  register double d31 __asm__("d31") = -31.5;
  __asm__ volatile("bl " ASM_SYM(__aarch64_ldadd4_acq_rel_preservemost)
                   : "+r"(x0), "+r"(x1), "+r"(x5), "+r"(x8), "+r"(x13),
                     "+r"(x14), "+w"(d0), "+w"(d31)
                   :
                   : "x2", "x3", "x4", "x15", "x16", "x17", "x30", "cc",
                     "memory");
  if (d0 != 1.25 || d31 != -31.5)
    return 5;
#  else
  __asm__ volatile(
      "bl " ASM_SYM(__aarch64_ldadd4_acq_rel_preservemost)
      : "+r"(x0), "+r"(x1), "+r"(x5), "+r"(x8), "+r"(x13), "+r"(x14)
      :
      : "x2", "x3", "x4", "x15", "x16", "x17", "x30", "cc", "memory");
#  endif
  if ((uint32_t)x0 != 40 || mem != 42)
    return 1;
  if (x5 != 0x0505050550505050ULL || x8 != 0x0808080880808080ULL ||
      x13 != 0x0d0d0d0dd0d0d0d0ULL || x14 != 0x0e0e0e0ee0e0e0e0ULL)
    return 2;
  return 0;
}

// Test 16-byte compare-and-swap, which uses the full x0-x4 argument/return
// footprint (expected in x0:x1, desired in x2:x3, pointer in x4) and verify
// that x5, x13, and x14 remain intact (unlike libgcc's `__aarch64_cas16_*`,
// which clobbers x13-x14).
static int check_cas16_preservemost(void) {
  __attribute__((aligned(16))) uint64_t pair[2] = {11, 22};
  register uint64_t x0 __asm__("x0") = 11;
  register uint64_t x1 __asm__("x1") = 22;
  register uint64_t x2 __asm__("x2") = 33;
  register uint64_t x3 __asm__("x3") = 44;
  register uint64_t *x4 __asm__("x4") = pair;
  register uint64_t x5 __asm__("x5") = 0x5555aaaa5555aaaaULL;
  register uint64_t x13 __asm__("x13") = 0xdddd2222dddd2222ULL;
  register uint64_t x14 __asm__("x14") = 0xeeee1111eeee1111ULL;
  __asm__ volatile("bl " ASM_SYM(__aarch64_cas16_acq_rel_preservemost)
                   : "+r"(x0), "+r"(x1), "+r"(x2), "+r"(x3), "+r"(x4), "+r"(x5),
                     "+r"(x13), "+r"(x14)
                   :
                   : "x15", "x16", "x17", "x30", "cc", "memory");
  if (x0 != 11 || x1 != 22 || pair[0] != 33 || pair[1] != 44)
    return 3;
  if (x5 != 0x5555aaaa5555aaaaULL || x13 != 0xdddd2222dddd2222ULL ||
      x14 != 0xeeee1111eeee1111ULL)
    return 4;
  return 0;
}

#endif

int main(void) {
#if defined(__aarch64__) && !defined(__arm64ec__)
  int rc = check_ldadd4_preservemost();
  if (rc != 0)
    return rc;
  return check_cas16_preservemost();
#else
  return 0;
#endif
}
