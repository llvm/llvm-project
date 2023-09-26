//===-- Implementation of the GPU remquof function ------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/math/remquof.h"
#include "src/__support/common.h"

#include "common.h"

namespace LIBC_NAMESPACE {

LLVM_LIBC_FUNCTION(float, remquof, (float x, float y, int *quo)) {
  return internal::remquof(x, y, quo);
}

} // namespace LIBC_NAMESPACE
