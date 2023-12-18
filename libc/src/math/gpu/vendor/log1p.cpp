//===-- Implementation of the GPU log1p function --------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/math/log1p.h"
#include "src/__support/common.h"

#include "common.h"

namespace LIBC_NAMESPACE {

LLVM_LIBC_FUNCTION(double, log1p, (double x)) { return internal::log1p(x); }

} // namespace LIBC_NAMESPACE
