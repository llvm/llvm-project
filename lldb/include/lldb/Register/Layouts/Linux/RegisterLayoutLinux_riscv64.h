//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_REGISTER_LAYOUTS_LINUX_REGISTERLAYOUTLINUX_RISCV64_H
#define LLDB_REGISTER_LAYOUTS_LINUX_REGISTERLAYOUTLINUX_RISCV64_H

#include "lldb/Register/RegisterLayout.h"
#include "lldb/Utility/Flags.h"

namespace lldb_private {

class ArchSpec;

class RegisterLayoutLinux_riscv64 final : public RegisterLayout {
public:
  // Mirrors Linux's user_regs_struct and double-precision floating-point
  // regset. Explicit alignment keeps the format independent of the reader's
  // host ABI.
  struct alignas(8) GPR {
    uint64_t gpr[32];
  };

  struct alignas(8) FPR {
    uint64_t fpr[32];
    uint32_t fcsr;
  };
  static_assert(sizeof(GPR) == 256);
  static_assert(sizeof(FPR) == 264);

  explicit RegisterLayoutLinux_riscv64(const ArchSpec &arch, Flags features);

  RegisterLayoutLinux_riscv64() = delete;
  RegisterLayoutLinux_riscv64(const RegisterLayoutLinux_riscv64 &) = default;
  RegisterLayoutLinux_riscv64(RegisterLayoutLinux_riscv64 &&) = delete;
};

} // namespace lldb_private

#endif
