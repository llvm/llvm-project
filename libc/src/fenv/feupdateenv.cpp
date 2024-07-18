//===-- Implementation of feupdateenv function ----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/fenv/feupdateenv.h"
#include "src/__support/FPUtil/FEnvImpl.h"
#include "src/__support/common.h"

#include "hdr/types/fenv_t.h"
#include "src/__support/macros/config.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(int, feupdateenv, (const fenv_t *envp)) {
  int current_excepts = fputil::test_except(FE_ALL_EXCEPT);
  if (fputil::set_env(envp) != 0)
    return -1;
  return fputil::raise_except(current_excepts);
}

} // namespace LIBC_NAMESPACE_DECL
