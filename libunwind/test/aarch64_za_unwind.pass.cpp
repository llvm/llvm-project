//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: linux && target={{aarch64-.+}}

#include <libunwind.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/auxv.h>

// Basic test of unwinding with SME lazy saves. This tests libunwind disables ZA
// (and commits a lazy save of ZA) before resuming from unwinding.

// Note: This test requires SME (and is setup to pass on targets without SME).

static bool checkHasSME() {
  constexpr int hwcap2_sme = (1 << 23);
  unsigned long hwcap2 = getauxval(AT_HWCAP2);
  return (hwcap2 & hwcap2_sme) != 0;
}

struct TPIDR2Block {
  void *za_save_buffer;
  uint64_t num_save_slices;
};

__attribute__((noinline)) void private_za() {
  // Note: Lazy save active on entry to function.
  unw_context_t context;
  unw_cursor_t cursor;

  unw_getcontext(&context);
  unw_init_local(&cursor, &context);
  unw_step(&cursor);
  unw_resume(&cursor);
}

bool isZAOn() {
  register uint64_t svcr asm("x20");
  asm(".inst 0xd53b4254" : "=r"(svcr));
  return (svcr & 0b10) != 0;
}

__attribute__((noinline)) void za_function_with_lazy_save() {
  register uint64_t tmp asm("x8");

  // SMSTART ZA (should zero ZA)
  asm(".inst 0xd503457f");

  // RDSVL x8, #1 (read streaming vector length)
  asm(".inst 0x04bf5828" : "=r"(tmp));

  // Allocate and fill ZA save buffer with 0xAA.
  size_t buffer_size = tmp * tmp;
  uint8_t *za_save_buffer = (uint8_t *)alloca(buffer_size);
  memset(za_save_buffer, 0xAA, buffer_size);

  TPIDR2Block block = {za_save_buffer, tmp};
  tmp = reinterpret_cast<uint64_t>(&block);

  // MRS TPIDR2_EL0, x8 (setup lazy save of ZA)
  asm(".inst 0xd51bd0a8" ::"r"(tmp));

  // ZA should be on before unwinding.
  if (!isZAOn()) {
    fprintf(stderr, __FILE__ ": fail (ZA not on before call)\n");
    abort();
  } else {
    fprintf(stderr, __FILE__ ": pass (ZA on before call)\n");
  }

  private_za();

  // ZA should be off after unwinding.
  if (isZAOn()) {
    fprintf(stderr, __FILE__ ": fail (ZA on after unwinding)\n");
    abort();
  } else {
    fprintf(stderr, __FILE__ ": pass (ZA off after unwinding)\n");
  }

  // MRS x8, TPIDR2_EL0 (read TPIDR2_EL0)
  asm(".inst 0xd53bd0a8" : "=r"(tmp));
  // ZA should have been saved (TPIDR2_EL0 zero).
  if (tmp != 0) {
    fprintf(stderr, __FILE__ ": fail (TPIDR2_EL0 non-null after unwinding)\n");
    abort();
  } else {
    fprintf(stderr, __FILE__ ": pass (TPIDR2_EL0 null after unwinding)\n");
  }

  // ZA (all zero) should have been saved to the buffer.
  for (unsigned i = 0; i < buffer_size; ++i) {
    if (za_save_buffer[i] != 0) {
      fprintf(stderr,
              __FILE__ ": fail (za_save_buffer non-zero after unwinding)\n");
      abort();
    }
  }
  fprintf(stderr, __FILE__ ": pass (za_save_buffer zero'd after unwinding)\n");
}

int main(int, char **) {
  if (!checkHasSME()) {
    fprintf(stderr, __FILE__ ": pass (no SME support)\n");
    return 0; // Pass (SME is required for this test to run).
  }
  za_function_with_lazy_save();
  return 0;
}
