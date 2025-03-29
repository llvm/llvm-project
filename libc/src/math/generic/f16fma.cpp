//===-- Implementation of f16fma function ---------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/math/f16fma.h"
#include "src/__support/FPUtil/FMA.h"
#include "src/__support/common.h"
#include "src/__support/macros/config.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(float16, f16fma, (double x, double y, double z)) {
  return fputil::fma<float16>(x, y, z);
}

} // namespace LIBC_NAMESPACE_DECL
