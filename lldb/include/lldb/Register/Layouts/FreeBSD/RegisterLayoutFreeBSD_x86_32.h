//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_REGISTER_LAYOUTS_FREEBSD_REGISTERLAYOUTFREEBSD_X86_32_H
#define LLDB_REGISTER_LAYOUTS_FREEBSD_REGISTERLAYOUTFREEBSD_X86_32_H

#include "lldb/Register/Layouts/Common/RegisterLayout_x86.h"

namespace lldb_private {

class ArchSpec;

class RegisterLayoutFreeBSD_x86_32 final : public RegisterLayout_x86 {
public:
  explicit RegisterLayoutFreeBSD_x86_32(
      const ArchSpec &arch, RegisterLayout_x86::TagWordView tag_word =
                                RegisterLayout_x86::TagWordView::Full);

  /// Bind this 32-bit target to an explicitly selected native GPR format.
  RegisterLayoutFreeBSD_x86_32(const ArchSpec &arch, GPRMode mode,
                               TagWordView tag_word = TagWordView::Full);

  RegisterLayoutFreeBSD_x86_32() = delete;
  RegisterLayoutFreeBSD_x86_32(const RegisterLayoutFreeBSD_x86_32 &) = delete;
  RegisterLayoutFreeBSD_x86_32(RegisterLayoutFreeBSD_x86_32 &&) = delete;
};

} // namespace lldb_private

#endif
