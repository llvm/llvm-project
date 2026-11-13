//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_REGISTER_LAYOUTS_LINUX_REGISTERLAYOUTLINUX_LOONGARCH64_H
#define LLDB_REGISTER_LAYOUTS_LINUX_REGISTERLAYOUTLINUX_LOONGARCH64_H
#include "lldb/Register/Common/RegisterInfo_loongarch64.h"
#include "lldb/Register/RegisterLayout.h"
#include <cstdint>
namespace lldb_private {
namespace linux_loongarch64 {
struct alignas(8) GPR {
  uint64_t gpr[32];

  uint64_t orig_a0;
  uint64_t csr_era;
  uint64_t csr_badv;
  uint64_t reserved[10];
};

struct alignas(8) FPR {
  uint64_t fpr[32];
  uint64_t fcc;
  uint32_t fcsr;
};

/* 32 registers, 128 bits width per register. */
struct LSX {
  uint64_t vr[32 * 2];
};

/* 32 registers, 256 bits width per register. */
struct LASX {
  uint64_t xr[32 * 4];
};

static_assert(sizeof(GPR) == 360);
static_assert(sizeof(FPR) == 272);
static_assert(sizeof(LSX) == 512);
static_assert(sizeof(LASX) == 1024);
} // namespace linux_loongarch64
RegisterLayout CreateLinuxLoongArch64RegisterLayout(const ArchSpec &arch);
} // namespace lldb_private
#endif
