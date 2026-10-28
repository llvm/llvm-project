//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Definition of the asinhbf16 entrypoint. It forwards to the implementation
/// in src/__support/math/asinhbf16.h.
///
//===----------------------------------------------------------------------===//

#include "src/math/asinhbf16.h"
#include "src/__support/math/asinhbf16.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(bfloat16, asinhbf16, (bfloat16 x)) {
  return math::asinhbf16(x);
}

} // namespace LIBC_NAMESPACE_DECL
