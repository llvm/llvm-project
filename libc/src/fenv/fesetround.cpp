//===-- Implementation of fesetround function -----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/fenv/fesetround.h"
#include "src/__support/FPUtil/FEnvUtils.h"
#include "src/__support/common.h"

namespace __llvm_libc {

LLVM_LIBC_FUNCTION(int, fesetround, (int m)) { return fputil::setRound(m); }

} // namespace __llvm_libc
