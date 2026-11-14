//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Implementation of the iscanonicalf80 function.
///
//===----------------------------------------------------------------------===//

#include "src/math/iscanonicalf80.h"
#include "src/__support/CPP/bit.h"
#include "src/__support/math/iscanonicalf80.h"

namespace LIBC_NAMESPACE_DECL {

using LIBC_NAMESPACE::fputil::Float80;

LLVM_LIBC_FUNCTION(int, iscanonicalf80, (float80 x)) {
  return math::iscanonicalf80(cpp::bit_cast<Float80>(x));
}

} // namespace LIBC_NAMESPACE_DECL
