//===-- Implementation of the GPU coshf function --------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/math/coshf.h"
#include "src/__support/math/coshf.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(float, coshf, (float x)) { return __builtin_coshf(x); }

} // namespace LIBC_NAMESPACE_DECL
