//===-- Implementation of feholdexcept function ---------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/fenv/feholdexcept.h"
#include "hdr/types/fenv_t.h"
#include "src/__support/FPUtil/FEnvImpl.h"
#include "src/__support/common.h"
#include "src/__support/macros/config.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(int, feholdexcept, (fenv_t * envp)) {
  if (fputil::get_env(envp) != 0)
    return -1;
  fputil::clear_except(FE_ALL_EXCEPT);
  fputil::disable_except(FE_ALL_EXCEPT);
  return 0;
}

} // namespace LIBC_NAMESPACE_DECL
