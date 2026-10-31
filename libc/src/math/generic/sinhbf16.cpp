//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Implementation for sinhbf16(x) function.
///
//===----------------------------------------------------------------------===//

#include "src/math/sinhbf16.h"
#include "src/__support/math/sinhbf16.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(bfloat16, sinhbf16, (bfloat16 x)) {
  return math::sinhbf16(x);
}

} // namespace LIBC_NAMESPACE_DECL
