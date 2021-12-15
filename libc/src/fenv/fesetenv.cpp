//===-- Implementation of fesetenv function -----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/fenv/fesetenv.h"
#include "src/__support/FPUtil/FEnvUtils.h"
#include "src/__support/common.h"

namespace __llvm_libc {

LLVM_LIBC_FUNCTION(int, fesetenv, (const fenv_t *envp)) {
  return fputil::set_env(envp);
}

} // namespace __llvm_libc
