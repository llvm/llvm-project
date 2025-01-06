//===-- Extended-precision atan2 function ---------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/math/atan2l.h"
#include "src/__support/common.h"
#include "src/__support/macros/properties/types.h"
#include "src/math/atan2.h"

namespace LIBC_NAMESPACE_DECL {

// TODO: Implement this for extended precision.
LLVM_LIBC_FUNCTION(long double, atan2l, (long double y, long double x)) {
#if defined(LIBC_TYPES_LONG_DOUBLE_IS_FLOAT64)
  return static_cast<long double>(
      atan2(static_cast<double>(y), static_cast<double>(x)));
#else
#error "Extended precision is not yet supported"
#endif
}

} // namespace LIBC_NAMESPACE_DECL
