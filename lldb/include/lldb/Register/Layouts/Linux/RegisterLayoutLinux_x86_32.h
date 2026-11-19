//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_REGISTER_LAYOUTS_LINUX_REGISTERLAYOUTLINUX_X86_32_H
#define LLDB_REGISTER_LAYOUTS_LINUX_REGISTERLAYOUTLINUX_X86_32_H

#include "lldb/Register/Layouts/Common/RegisterLayout_x86.h"

namespace lldb_private {

class ArchSpec;

class RegisterLayoutLinux_x86_32 final : public RegisterLayout_x86 {
public:
  explicit RegisterLayoutLinux_x86_32(
      const ArchSpec &arch, RegisterLayout_x86::TagWordView tag_word =
                                RegisterLayout_x86::TagWordView::Full);

  /// Bind this 32-bit target to an explicitly selected native GPR format.
  RegisterLayoutLinux_x86_32(const ArchSpec &arch, GPRMode mode,
                             TagWordView tag_word = TagWordView::Full);

  /// Offset of the kernel's original syscall number in the saved GPR bank.
  static uint32_t GetOriginalSyscallNumberOffset();

  RegisterLayoutLinux_x86_32() = delete;
  RegisterLayoutLinux_x86_32(const RegisterLayoutLinux_x86_32 &) = delete;
  RegisterLayoutLinux_x86_32(RegisterLayoutLinux_x86_32 &&) = delete;
};

} // namespace lldb_private

#endif
