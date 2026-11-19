//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Entrypoint definition for the coshbf16 function.
///
//===----------------------------------------------------------------------===//

#include "src/__support/math/coshbf16.h"
#include "src/math/coshbf16.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(bfloat16, coshbf16, (bfloat16 x)) {
  return math::coshbf16(x);
}

} // namespace LIBC_NAMESPACE_DECL
