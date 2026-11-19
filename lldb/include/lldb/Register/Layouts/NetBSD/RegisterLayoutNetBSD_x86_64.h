//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_REGISTER_LAYOUTS_NETBSD_REGISTERLAYOUTNETBSD_X86_64_H
#define LLDB_REGISTER_LAYOUTS_NETBSD_REGISTERLAYOUTNETBSD_X86_64_H

#include "lldb/Register/Layouts/Common/RegisterLayout_x86.h"

namespace lldb_private {

class ArchSpec;

class RegisterLayoutNetBSD_x86_64 final : public RegisterLayout_x86 {
public:
  explicit RegisterLayoutNetBSD_x86_64(
      const ArchSpec &arch, RegisterLayout_x86::TagWordView tag_word =
                                RegisterLayout_x86::TagWordView::Full);

  /// Native 64-bit GPR slots, also usable by compatibility-mode targets.
  static uint32_t GetGPRSize();
  static std::optional<RegisterPiece> GetGPRField(RegisterID id);

  RegisterLayoutNetBSD_x86_64() = delete;
  RegisterLayoutNetBSD_x86_64(const RegisterLayoutNetBSD_x86_64 &) = delete;
  RegisterLayoutNetBSD_x86_64(RegisterLayoutNetBSD_x86_64 &&) = delete;
};

} // namespace lldb_private

#endif
