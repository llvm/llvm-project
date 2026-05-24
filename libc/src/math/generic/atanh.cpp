//===-- Double-precision atanh implementation -----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/math/atanh.h"
#include "src/__support/math/atanh.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(double, atanh, (double x)) { return math::atanh(x); }

} // namespace LIBC_NAMESPACE_DECL
