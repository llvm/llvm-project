// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: target={{mips.*}}

// The o32 context is sized for 32 double-width float slots even when the FPU
// only holds 32-bit values, so that one context size covers every o32 FPU
// configuration. Registers.hpp asserts the base offset and stride that
// UnwindRegisters{Save,Restore}.S depend on; this is the outer size check.

#include "../src/config.h"

#if defined(_LIBUNWIND_TARGET_MIPS_O32) && defined(__mips_hard_float)

#include "../src/Registers.hpp"

using namespace libunwind;

static_assert(sizeof(Registers_mips_o32) == 4 * 36 + 8 * 32,
              "the o32 context layout no longer matches the offsets hard-coded "
              "in UnwindRegistersSave.S and UnwindRegistersRestore.S");

#endif // _LIBUNWIND_TARGET_MIPS_O32 && __mips_hard_float
