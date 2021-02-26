//===-- Implementation of fegetenv function -----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/fenv/fegetenv.h"
#include "src/__support/common.h"
#include "utils/FPUtil/FEnv.h"

namespace __llvm_libc {

LLVM_LIBC_FUNCTION(int, fegetenv, (fenv_t * envp)) {
  return fputil::getEnv(envp);
}

} // namespace __llvm_libc
