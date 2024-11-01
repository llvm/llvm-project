//===-- Implementation of the GPU scalbnf function ------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/math/scalbnf.h"
#include "src/__support/common.h"

#include "common.h"

namespace __llvm_libc {

LLVM_LIBC_FUNCTION(float, scalbnf, (float x, int y)) {
  return internal::scalbnf(x, y);
}

} // namespace __llvm_libc
