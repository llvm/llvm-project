//===-- Linux implementation of the cnd_broadcast function ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "CndVar.h"

#include "src/threads/cnd_broadcast.h"
#include "src/__support/common.h"

namespace __llvm_libc {

LLVM_LIBC_FUNCTION(int, cnd_broadcast, (cnd_t * cond)) {
  CndVar *cndvar = reinterpret_cast<CndVar *>(cond);
  return cndvar->broadcast();
}

} // namespace __llvm_libc
