// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: target={{mips.*}}

// Exercise the o32 single-float register accessors. Before single float was
// supported these paths either reported the floating point registers as
// invalid or aborted with "mips_o32 float support not implemented".

#include "../src/config.h"

#if defined(_LIBUNWIND_TARGET_MIPS_O32) && defined(__mips_hard_float) &&       \
    defined(__mips_single_float)

#include "../src/Registers.hpp"

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

using namespace libunwind;

int main(int, char **) {
  Registers_mips_o32 regs;

  for (int n = UNW_MIPS_F0; n <= UNW_MIPS_F31; ++n) {
    if (!regs.validRegister(n))
      abort();
    if (!regs.validFloatRegister(n))
      abort();
  }

  // Under single float each register is independent, with no even/odd pairing,
  // so a distinct value per register has to survive a round trip.
  for (int n = UNW_MIPS_F0; n <= UNW_MIPS_F31; ++n)
    regs.setRegister(n, 0xf0000000u + static_cast<uint32_t>(n - UNW_MIPS_F0));
  for (int n = UNW_MIPS_F0; n <= UNW_MIPS_F31; ++n) {
    if (regs.getRegister(n) !=
        0xf0000000u + static_cast<uint32_t>(n - UNW_MIPS_F0))
      abort();
  }

  // 2.5 is exactly representable in single precision, so this round trip is
  // not subject to rounding.
  regs.setFloatRegister(UNW_MIPS_F7, 2.5);
  if (regs.getFloatRegister(UNW_MIPS_F7) != 2.5)
    abort();

  // The raw and floating point views address the same storage.
  float single = 2.5f;
  uint32_t bits;
  memcpy(&bits, &single, sizeof(bits));
  if (regs.getRegister(UNW_MIPS_F7) != bits)
    abort();

  return 0;
}

#else
int main(int, char **) { return 0; }
#endif
