//===-- Implementation of the GPU exp2f function --------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/math/exp2f.h"
#include "src/__support/common.h" // for LLVM_LIBC_FUNCTION
#include "src/__support/macros/config.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(float, exp2f, (float x)) { return __builtin_exp2f(x); }

} // namespace LIBC_NAMESPACE_DECL
