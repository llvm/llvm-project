//===-- Implementation of feraiseexcept function --------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/__support/common.h"
#include "utils/FPUtil/FEnv.h"

namespace __llvm_libc {

int LLVM_LIBC_ENTRYPOINT(feraiseexcept)(int e) {
  return fputil::raiseExcept(e);
}

} // namespace __llvm_libc
