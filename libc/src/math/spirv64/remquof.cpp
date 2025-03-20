//===-- Implementation of the GPU remquof function ------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/math/remquof.h"
#include "src/__support/common.h"

#include "declarations.h"
#include "src/__support/macros/config.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(float, remquof, (float x, float y, int *quo)) {
  int tmp;
  float r = __ocml_remquo_f32(x, y, (gpu::Private<int> *)&tmp);
  *quo = tmp;
  return r;
}

} // namespace LIBC_NAMESPACE_DECL
