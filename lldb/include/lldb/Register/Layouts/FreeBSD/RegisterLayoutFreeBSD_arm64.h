//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_REGISTER_LAYOUTS_FREEBSD_REGISTERLAYOUTFREEBSD_ARM64_H
#define LLDB_REGISTER_LAYOUTS_FREEBSD_REGISTERLAYOUTFREEBSD_ARM64_H
#include "lldb/Register/Common/RegisterInfo_arm64.h"
#include "lldb/Register/RegisterLayout.h"
#include "lldb/Utility/ArchSpec.h"
namespace lldb_private {
namespace freebsd_arm64 {
// FreeBSD arm64 reg and fpreg, independent of the reader's host ABI.
struct GPR {
  uint64_t x[30];
  uint64_t lr;
  uint64_t sp;
  uint64_t elr;
  uint64_t spsr;
};
struct alignas(16) FPR {
  uint8_t q[32][16];
  uint32_t fpsr;
  uint32_t fpcr;
};
static_assert(sizeof(GPR) == 272);
static_assert(sizeof(FPR) == 528);
} // namespace freebsd_arm64

Arm64RegisterFeatures DecodeFreeBSDArm64RegisterFeatures(uint64_t hwcap,
                                                         uint64_t hwcap2);
RegisterLayout CreateFreeBSDArm64RegisterLayout(const ArchSpec &arch,
                                                uint64_t hwcap = 0,
                                                uint64_t hwcap2 = 0);
} // namespace lldb_private
#endif
