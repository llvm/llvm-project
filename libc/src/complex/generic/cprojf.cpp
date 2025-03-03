//===-- Implementation of cprojf function ---------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/complex/cprojf.h"
#include "src/__support/common.h"
#include "src/__support/complex_basic_ops.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(_Complex float, cprojf, (_Complex float x)) {
  return project<_Complex float>(x);
}

} // namespace LIBC_NAMESPACE_DECL
