//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_REGISTER_LAYOUTS_DARWIN_REGISTERLAYOUTDARWIN_ARM64_H
#define LLDB_REGISTER_LAYOUTS_DARWIN_REGISTERLAYOUTDARWIN_ARM64_H

#include "lldb/Register/RegisterLayout.h"

namespace lldb_private {

/// Saved Mach thread-state banks for arm64.
class RegisterLayoutDarwin_arm64 final : public RegisterLayout {
public:
  // mirrors <mach/arm/thread_status.h> arm_thread_state64_t
  struct GPR {
    uint64_t x[29]; // x0-x28
    uint64_t fp;    // x29
    uint64_t lr;    // x30
    uint64_t sp;    // x31
    uint64_t pc;    // pc
    uint32_t cpsr;  // cpsr
  };

  struct VReg {
    alignas(16) char bytes[16];
  };

  // mirrors <mach/arm/thread_status.h> arm_neon_state64_t
  struct FPU {
    VReg v[32];
    uint32_t fpsr;
    uint32_t fpcr;
  };

  // mirrors <mach/arm/thread_status.h> arm_exception_state64_t
  struct EXC {
    uint64_t far;       // Virtual Fault Address
    uint32_t esr;       // Exception syndrome
    uint32_t exception; // number of arm exception token
  };

  // mirrors <mach/arm/thread_status.h> arm_debug_state64_t
  struct DBG {
    uint64_t bvr[16];
    uint64_t bcr[16];
    uint64_t wvr[16];
    uint64_t wcr[16];
    uint64_t mdscr_el1;
  };

  enum {
    GPRRegSet = 6,  // ARM_THREAD_STATE64
    FPURegSet = 17, // ARM_NEON_STATE64
    EXCRegSet = 7,  // ARM_EXCEPTION_STATE64
    DBGRegSet = 15  // ARM_DEBUG_STATE64
  };

  RegisterLayoutDarwin_arm64();
  static const RegisterLayoutDarwin_arm64 &GetShared();
};

} // namespace lldb_private

#endif
