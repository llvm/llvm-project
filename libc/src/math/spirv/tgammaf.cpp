//===-- Implementation of the GPU tgammaf function ------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/math/tgammaf.h"
#include "src/__support/common.h"

#include "src/__support/macros/config.h"

namespace LIBC_NAMESPACE_DECL {

extern "C" float __attribute__((overloadable)) __spirv_ocl_tgamma(float);
LLVM_LIBC_FUNCTION(float, tgammaf, (float x)) { return __spirv_ocl_tgamma(x); }

} // namespace LIBC_NAMESPACE_DECL
