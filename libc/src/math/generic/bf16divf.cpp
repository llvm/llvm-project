//===-- Implementation of bf16divf function -------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/math/bf16divf.h"

#include "src/__support/FPUtil/bfloat16.h"
#include "src/__support/FPUtil/generic/div.h"
#include "src/__support/common.h"
#include "src/__support/macros/config.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(bfloat16, bf16divf, (float x, float y)) {
  return fputil::generic::div<bfloat16>(x, y);
}

} // namespace LIBC_NAMESPACE_DECL
