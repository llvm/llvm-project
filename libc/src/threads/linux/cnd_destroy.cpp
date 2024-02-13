//===-- Linux implementation of the cnd_destroy function ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "CndVar.h"

#include "src/threads/cnd_destroy.h"
#include "src/__support/common.h"

namespace LIBC_NAMESPACE {

LLVM_LIBC_FUNCTION(void, cnd_destroy, (cnd_t * cond)) {
  CndVar *cndvar = reinterpret_cast<CndVar *>(cond);
  CndVar::destroy(cndvar);
}

} // namespace LIBC_NAMESPACE
