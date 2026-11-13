//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_REGISTER_LAYOUTS_LINUX_REGISTERLAYOUTLINUX_RISCV64_H
#define LLDB_REGISTER_LAYOUTS_LINUX_REGISTERLAYOUTLINUX_RISCV64_H

#include "lldb/Register/Common/RegisterInfo_riscv64.h"
#include "lldb/Register/RegisterLayout.h"

namespace lldb_private {
namespace linux_riscv64 {
// Mirrors Linux's user_regs_struct and double-precision floating-point regset.
// Explicit alignment keeps the format independent of the reader's host ABI.
struct alignas(8) GPR {
  uint64_t gpr[32];
};
struct alignas(8) FPR {
  uint64_t fpr[32];
  uint32_t fcsr;
};
static_assert(sizeof(GPR) == 256);
static_assert(sizeof(FPR) == 264);
} // namespace linux_riscv64

RegisterLayout CreateLinuxRISCV64RegisterLayout(const ArchSpec &arch,
                                                Flags features);
} // namespace lldb_private

#endif
