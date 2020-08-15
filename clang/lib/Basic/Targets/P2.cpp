//===--- P2.cpp - Implement P2 target feature support -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements P2 TargetInfo objects.
//
//===----------------------------------------------------------------------===//

#include "P2.h"
#include "clang/Basic/MacroBuilder.h"
#include "llvm/ADT/StringSwitch.h"

using namespace clang;
using namespace clang::targets;

namespace clang {
namespace targets {

} // namespace targets
} // namespace clang

const char *const P2TargetInfo::GCCRegNames[] = {
    "r0", "r1", "r2",  "r3",  "r4",  "r5",  "r6",  "r7",
    "r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15",
    "r16", "r17", "r18", "r19", "r20", "r21", "r22", "r23",
    "r24", "r25", "r26", "r27", "r28", "r29", "r30", "r31",
    "ijmp3", "iret3", "ijmp2", "iret2", "ijmp1", "iret1", "pa", "pb",
    "ptra", "ptrb", "dira", "dirb", "outa", "outb", "ina", "inb"
};

ArrayRef<const char *> P2TargetInfo::getGCCRegNames() const {
  return llvm::makeArrayRef(GCCRegNames);
}

void P2TargetInfo::getTargetDefines(const LangOptions &Opts,
                                     MacroBuilder &Builder) const {
    Builder.defineMacro("__propeller2__");
    Builder.defineMacro("__p2llvm__");
}
