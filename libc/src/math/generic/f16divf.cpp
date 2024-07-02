//===-- Implementation of f16divf function --------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/math/f16divf.h"
#include "src/__support/FPUtil/generic/div.h"
#include "src/__support/common.h"

namespace LIBC_NAMESPACE {

LLVM_LIBC_FUNCTION(float16, f16divf, (float x, float y)) {
  return fputil::generic::div<float16>(x, y);
}

} // namespace LIBC_NAMESPACE
