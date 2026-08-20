//===-- Implementation of floorf function ---------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/math/floorf.h"
#include "src/__support/math/floorf.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(float, floorf, (float x)) { return math::floorf(x); }

} // namespace LIBC_NAMESPACE_DECL
