//===-- Implementation of the sinh function for GPU -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/math/sinh.h"
#include "src/__support/common.h"

#include "common.h"

namespace __llvm_libc {

LLVM_LIBC_FUNCTION(double, sinh, (double x)) { return internal::sinh(x); }

} // namespace __llvm_libc
