//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_REGISTER_LAYOUTS_DARWIN_REGISTERLAYOUTDARWIN_RISCV32_H
#define LLDB_REGISTER_LAYOUTS_DARWIN_REGISTERLAYOUTDARWIN_RISCV32_H

#include "lldb/Register/RegisterLayout.h"

namespace lldb_private {

/// Saved Mach thread-state banks for riscv32.
class RegisterLayoutDarwin_riscv32 final : public RegisterLayout {
public:
  struct GPR {
    uint32_t x0;
    uint32_t x1;
    uint32_t x2;
    uint32_t x3;
    uint32_t x4;
    uint32_t x5;
    uint32_t x6;
    uint32_t x7;
    uint32_t x8;
    uint32_t x9;
    uint32_t x10;
    uint32_t x11;
    uint32_t x12;
    uint32_t x13;
    uint32_t x14;
    uint32_t x15;
    uint32_t x16;
    uint32_t x17;
    uint32_t x18;
    uint32_t x19;
    uint32_t x20;
    uint32_t x21;
    uint32_t x22;
    uint32_t x23;
    uint32_t x24;
    uint32_t x25;
    uint32_t x26;
    uint32_t x27;
    uint32_t x28;
    uint32_t x29;
    uint32_t x30;
    uint32_t x31;
    uint32_t pc;
  };

  struct FPU {
    uint32_t f0;
    uint32_t f1;
    uint32_t f2;
    uint32_t f3;
    uint32_t f4;
    uint32_t f5;
    uint32_t f6;
    uint32_t f7;
    uint32_t f8;
    uint32_t f9;
    uint32_t f10;
    uint32_t f11;
    uint32_t f12;
    uint32_t f13;
    uint32_t f14;
    uint32_t f15;
    uint32_t f16;
    uint32_t f17;
    uint32_t f18;
    uint32_t f19;
    uint32_t f20;
    uint32_t f21;
    uint32_t f22;
    uint32_t f23;
    uint32_t f24;
    uint32_t f25;
    uint32_t f26;
    uint32_t f27;
    uint32_t f28;
    uint32_t f29;
    uint32_t f30;
    uint32_t f31;
    uint32_t fcsr;
  };

  struct EXC {
    uint32_t exception;
    uint32_t fsr;
    uint32_t far;
  };

  struct CSR {
    uint32_t csr[1024];
  };

  enum {
    GPRRegSet = 2,  // RV32_THREAD_STATE
    EXCRegSet = 3,  // RV32_EXCEPTION_STATE
    FPURegSet = 4,  // RV_FP32_STATE
    CSRRegSet1 = 6, // RV_CSR_STATE1
    CSRRegSet2 = 7, // RV_CSR_STATE2
    CSRRegSet3 = 8, // RV_CSR_STATE3
    CSRRegSet4 = 9, // RV_CSR_STATE4
    CSRRegSet = 10  // full 16kbyte CSR reg bank
  };

  RegisterLayoutDarwin_riscv32();
  static const RegisterLayoutDarwin_riscv32 &GetShared();
};

} // namespace lldb_private

#endif
