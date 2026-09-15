//===-- Implementation of ilogbf128 function ------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Implementation of the float128 ilogb function.
///
//===----------------------------------------------------------------------===//

#include "src/math/ilogbf128.h"
#include "src/__support/CPP/bit.h"
#include "src/__support/math/ilogbf128.h"

namespace LIBC_NAMESPACE_DECL {

using LIBC_NAMESPACE::fputil::Float128;

LLVM_LIBC_FUNCTION(int, ilogbf128, (float128 x)) {
  return math::ilogbf128(cpp::bit_cast<Float128>(x));
}

} // namespace LIBC_NAMESPACE_DECL
