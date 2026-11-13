//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_REGISTER_LAYOUTS_LINUX_REGISTERLAYOUTLINUX_ARM_H
#define LLDB_REGISTER_LAYOUTS_LINUX_REGISTERLAYOUTLINUX_ARM_H
#include "lldb/Register/Common/RegisterInfo_arm.h"
#include "lldb/Register/RegisterLayout.h"
#include "lldb/Utility/ArchSpec.h"
namespace lldb_private {
namespace linux_arm {
struct GPR {
  uint32_t r[16];   // R0-R15
  uint32_t cpsr;    // CPSR
  uint32_t orig_r0; // Linux preserves the syscall argument after CPSR.
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
struct TLS {
  uint32_t tpidruro;
};
} // namespace linux_arm
RegisterLayout CreateLinuxArmRegisterLayout(const ArchSpec &arch,
                                            bool tls = false);
} // namespace lldb_private
#endif
