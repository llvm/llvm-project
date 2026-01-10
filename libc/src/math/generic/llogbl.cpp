//===-- Implementation of llogbl function ---------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/math/llogbl.h"
#include "src/__support/common.h"

namespace LIBC_NAMESPACE_DECL {

// Export the public C symbol by wrapping the inline constexpr definition.
// This maintains binary compatibility with the shipped libc while allowing
// callers to evaluate llogbl at compile time or have it inlined.
LLVM_LIBC_FUNCTION(long, llogbl, (long double x)) {
  return LIBC_NAMESPACE::llogbl(x);
}

} // namespace LIBC_NAMESPACE_DECL
