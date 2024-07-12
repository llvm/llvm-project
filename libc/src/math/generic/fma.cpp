//===-- Implementation of fma function ------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/math/fma.h"
#include "src/__support/common.h"

#include "src/__support/FPUtil/FMA.h"

namespace LIBC_NAMESPACE {

LLVM_LIBC_FUNCTION(double, fma, (double x, double y, double z)) {
  return fputil::fma<double>(x, y, z);
}

} // namespace LIBC_NAMESPACE
