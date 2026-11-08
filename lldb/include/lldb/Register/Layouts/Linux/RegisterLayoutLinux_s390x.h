//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_REGISTER_LAYOUTS_LINUX_REGISTERLAYOUTLINUX_S390X_H
#define LLDB_REGISTER_LAYOUTS_LINUX_REGISTERLAYOUTLINUX_S390X_H
#include "lldb/Register/Common/RegisterInfo_s390x.h"
#include "lldb/Register/RegisterLayout.h"

namespace lldb_private {
namespace linux_s390x {
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
} // namespace linux_s390x

RegisterLayout CreateLinuxS390XRegisterLayout(const ArchSpec &arch);
} // namespace lldb_private
#endif
