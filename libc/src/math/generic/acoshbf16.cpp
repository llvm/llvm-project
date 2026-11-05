//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains the implementation of the acoshbf16 function.
///
//===----------------------------------------------------------------------===//

#include "src/math/acoshbf16.h"
#include "src/__support/math/acoshbf16.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(bfloat16, acoshbf16, (bfloat16 x)) {
  return math::acoshbf16(x);
}

} // namespace LIBC_NAMESPACE_DECL
