//===-- Implementation of canonicalize function ----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/math/canonicalize.h"
#include "src/__support/FPUtil/NearestIntegerOperations.h"
#include "src/__support/common.h"

namespace LIBC_NAMESPACE {

LLVM_LIBC_FUNCTION(int, canonicalize, (double *cx, const double *x)) {
    if (cx == nullptr || x == nullptr || std::isnan(*x) || std::isinf(*x))
        return 1;
    *cx = *x;
    return 0;
}

} // namespace LIBC_NAMESPACE
