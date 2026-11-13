//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_REGISTER_LAYOUTS_LINUX_REGISTERLAYOUTLINUX_PPC64LE_H
#define LLDB_REGISTER_LAYOUTS_LINUX_REGISTERLAYOUTLINUX_PPC64LE_H
#include "lldb/Register/Common/RegisterInfo_ppc64le.h"
#include "lldb/Register/RegisterLayout.h"
#include <cstdint>
namespace lldb_private {
namespace linux_ppc64le {
typedef struct _GPR {
  uint64_t r0;
  uint64_t r1;
  uint64_t r2;
  uint64_t r3;
  uint64_t r4;
  uint64_t r5;
  uint64_t r6;
  uint64_t r7;
  uint64_t r8;
  uint64_t r9;
  uint64_t r10;
  uint64_t r11;
  uint64_t r12;
  uint64_t r13;
  uint64_t r14;
  uint64_t r15;
  uint64_t r16;
  uint64_t r17;
  uint64_t r18;
  uint64_t r19;
  uint64_t r20;
  uint64_t r21;
  uint64_t r22;
  uint64_t r23;
  uint64_t r24;
  uint64_t r25;
  uint64_t r26;
  uint64_t r27;
  uint64_t r28;
  uint64_t r29;
  uint64_t r30;
  uint64_t r31;
  uint64_t pc;
  uint64_t msr;
  uint64_t origr3;
  uint64_t ctr;
  uint64_t lr;
  uint64_t xer;
  uint64_t cr;
  uint64_t softe;
  uint64_t trap;
  uint64_t pad[3];
} GPR;

typedef struct _FPR {
  uint64_t f0;
  uint64_t f1;
  uint64_t f2;
  uint64_t f3;
  uint64_t f4;
  uint64_t f5;
  uint64_t f6;
  uint64_t f7;
  uint64_t f8;
  uint64_t f9;
  uint64_t f10;
  uint64_t f11;
  uint64_t f12;
  uint64_t f13;
  uint64_t f14;
  uint64_t f15;
  uint64_t f16;
  uint64_t f17;
  uint64_t f18;
  uint64_t f19;
  uint64_t f20;
  uint64_t f21;
  uint64_t f22;
  uint64_t f23;
  uint64_t f24;
  uint64_t f25;
  uint64_t f26;
  uint64_t f27;
  uint64_t f28;
  uint64_t f29;
  uint64_t f30;
  uint64_t f31;
  uint64_t fpscr;
} FPR;

typedef struct _VMX {
  uint32_t vr0[4];
  uint32_t vr1[4];
  uint32_t vr2[4];
  uint32_t vr3[4];
  uint32_t vr4[4];
  uint32_t vr5[4];
  uint32_t vr6[4];
  uint32_t vr7[4];
  uint32_t vr8[4];
  uint32_t vr9[4];
  uint32_t vr10[4];
  uint32_t vr11[4];
  uint32_t vr12[4];
  uint32_t vr13[4];
  uint32_t vr14[4];
  uint32_t vr15[4];
  uint32_t vr16[4];
  uint32_t vr17[4];
  uint32_t vr18[4];
  uint32_t vr19[4];
  uint32_t vr20[4];
  uint32_t vr21[4];
  uint32_t vr22[4];
  uint32_t vr23[4];
  uint32_t vr24[4];
  uint32_t vr25[4];
  uint32_t vr26[4];
  uint32_t vr27[4];
  uint32_t vr28[4];
  uint32_t vr29[4];
  uint32_t vr30[4];
  uint32_t vr31[4];
  uint32_t pad[2];
  uint32_t vscr[2];
  uint32_t vrsave;
} VMX;

struct VSX {
  // NT_PPC_VSX stores only the extra 64-bit halves of VSR0-31.
  uint64_t upper[32];
};
} // namespace linux_ppc64le
RegisterLayout CreateLinuxPPC64LERegisterLayout(const ArchSpec &arch);
} // namespace lldb_private
#endif
