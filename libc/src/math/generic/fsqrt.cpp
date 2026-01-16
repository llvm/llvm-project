//===-- Implementation of fsqrt function ----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/math/fsqrt.h"
#include "src/__support/math/fsqrt.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(float, fsqrt, (double x)) { return math::fsqrt(x); }

} // namespace LIBC_NAMESPACE_DECL
