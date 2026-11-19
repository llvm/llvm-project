//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_REGISTER_LAYOUTS_DARWIN_REGISTERLAYOUTDARWIN_ARM32_H
#define LLDB_REGISTER_LAYOUTS_DARWIN_REGISTERLAYOUTDARWIN_ARM32_H

#include "lldb/Register/RegisterLayout.h"

namespace lldb_private {

/// Saved Mach thread-state banks for arm32.
class RegisterLayoutDarwin_arm32 final : public RegisterLayout {
public:
  struct GPR {
    uint32_t r[16]; // R0-R15
    uint32_t cpsr;  // CPSR
  };

  struct QReg {
    uint8_t bytes[16];
  };

  struct FPU {
    union {
      uint32_t s[32];
      uint64_t d[32];
      QReg q[16]; // the 128-bit NEON registers
    } floats;
    uint32_t fpscr;
  };

  struct EXC {
    uint32_t exception;
    uint32_t fsr; /* Fault status */
    uint32_t far; /* Virtual Fault Address */
  };

  struct DBG {
    uint32_t bvr[16];
    uint32_t bcr[16];
    uint32_t wvr[16];
    uint32_t wcr[16];
  };

  enum {
    GPRRegSet = 1,    // ARM_THREAD_STATE
    GPRAltRegSet = 9, // ARM_THREAD_STATE32
    FPURegSet = 2,    // ARM_VFP_STATE
    EXCRegSet = 3,    // ARM_EXCEPTION_STATE
    DBGRegSet = 4     // ARM_DEBUG_STATE
  };

  RegisterLayoutDarwin_arm32();
  static const RegisterLayoutDarwin_arm32 &GetShared();
};

} // namespace lldb_private

#endif
