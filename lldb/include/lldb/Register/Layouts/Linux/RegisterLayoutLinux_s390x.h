//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_REGISTER_LAYOUTS_LINUX_REGISTERLAYOUTLINUX_S390X_H
#define LLDB_REGISTER_LAYOUTS_LINUX_REGISTERLAYOUTLINUX_S390X_H

#include "lldb/Register/RegisterLayout.h"

namespace lldb_private {

class RegisterLayoutLinux_s390x final : public RegisterLayout {
public:
  struct alignas(8) GPR {
    uint64_t pswm;
    uint64_t pswa;
    uint64_t gpr[16];
    uint32_t acr[16];
    uint64_t orig_r2;
  };

  struct alignas(8) FPR {
    uint32_t fpc;
    uint32_t pad;
    uint64_t fpr[16];
  };

  struct LastBreak {
    uint64_t last_break;
  };

  struct SystemCall {
    uint32_t system_call;
  };

  static_assert(sizeof(GPR) == 216);
  static_assert(sizeof(FPR) == 136);
  static_assert(sizeof(LastBreak) == 8);
  static_assert(sizeof(SystemCall) == 4);

  explicit RegisterLayoutLinux_s390x(const ArchSpec &arch);

  RegisterLayoutLinux_s390x() = delete;
  RegisterLayoutLinux_s390x(const RegisterLayoutLinux_s390x &) = delete;
  RegisterLayoutLinux_s390x(RegisterLayoutLinux_s390x &&) = delete;
};

} // namespace lldb_private

#endif
