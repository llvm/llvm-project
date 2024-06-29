//===-- Single-precision fmodl function -----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/math/fmodl.h"
#include "src/__support/FPUtil/generic/FMod.h"
#include "src/__support/common.h"

namespace LIBC_NAMESPACE {

LLVM_LIBC_FUNCTION(long double, fmodl, (long double x, long double y)) {
  return fputil::generic::FMod<long double>::eval(x, y);
}

} // namespace LIBC_NAMESPACE
