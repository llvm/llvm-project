//===-- Implementation of ffma function -----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/math/ffma.h"
#include "src/__support/math/ffma.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(float, ffma, (double x, double y, double z)) {
  return math::ffma(x, y, z);
}

} // namespace LIBC_NAMESPACE_DECL
