//===-- Implementation of nanf16 function ---------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/math/nanf16.h"
#include "src/__support/math/nanf16.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(float16, nanf16, (const char *arg)) {
  return math::nanf16(arg);
}

} // namespace LIBC_NAMESPACE_DECL
