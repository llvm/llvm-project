//===-- Extended-precision atan2 function ---------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/math/atan2l.h"
#include "src/__support/FPUtil/cast.h"
#include "src/__support/common.h"
#include "src/__support/macros/properties/types.h"
#include "src/__support/math/atan2.h"
#include "src/__support/math/atan2f128.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(long double, atan2l, (long double y, long double x)) {
#if defined(LIBC_TYPES_LONG_DOUBLE_IS_FLOAT64)
  return static_cast<long double>(
      math::atan2(static_cast<double>(y), static_cast<double>(x)));
#elif defined(LIBC_TYPES_HAS_FLOAT128)
  // TODO: Once we have a software implementation of float128,
  // we can use it here unconditionally, even if float128 is not
  // available as a host type.
  return fputil::cast<long double>(
      math::atan2f128(fputil::cast<float128>(y), fputil::cast<float128>(x)));
#else
#error "Extended precision is not yet supported"
#endif
}

} // namespace LIBC_NAMESPACE_DECL
