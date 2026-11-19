//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_REGISTER_LAYOUTS_DARWIN_REGISTERLAYOUTDARWIN_X86_64_H
#define LLDB_REGISTER_LAYOUTS_DARWIN_REGISTERLAYOUTDARWIN_X86_64_H

#include "lldb/Register/RegisterLayout.h"

namespace lldb_private {

/// Saved Mach thread-state banks for x86_64.
class RegisterLayoutDarwin_x86_64 final : public RegisterLayout {
public:
  struct GPR {
    uint64_t rax;
    uint64_t rbx;
    uint64_t rcx;
    uint64_t rdx;
    uint64_t rdi;
    uint64_t rsi;
    uint64_t rbp;
    uint64_t rsp;
    uint64_t r8;
    uint64_t r9;
    uint64_t r10;
    uint64_t r11;
    uint64_t r12;
    uint64_t r13;
    uint64_t r14;
    uint64_t r15;
    uint64_t rip;
    uint64_t rflags;
    uint64_t cs;
    uint64_t fs;
    uint64_t gs;
  };

  struct MMSReg {
    uint8_t bytes[10];
    uint8_t pad[6];
  };

  struct XMMReg {
    uint8_t bytes[16];
  };

  struct FPU {
    uint32_t pad[2];
    uint16_t fcw; // "fctrl"
    uint16_t fsw; // "fstat"
    uint8_t ftw;  // "ftag"
    uint8_t pad1;
    uint16_t fop; // "fop"
    uint32_t ip;  // "fioff"
    uint16_t cs;  // "fiseg"
    uint16_t pad2;
    uint32_t dp; // "fooff"
    uint16_t ds; // "foseg"
    uint16_t pad3;
    uint32_t mxcsr;
    uint32_t mxcsrmask;
    MMSReg stmm[8];
    XMMReg xmm[16];
    uint8_t pad4[6 * 16];
    int pad5;
  };

  struct EXC {
    uint32_t trapno;
    uint32_t err;
    uint64_t faultvaddr;
  };

  enum { GPRRegSet = 4, FPURegSet = 5, EXCRegSet = 6 };

  RegisterLayoutDarwin_x86_64();
  static const RegisterLayoutDarwin_x86_64 &GetShared();
  static bool IsXMM(RegisterID id);
};

} // namespace lldb_private

#endif
