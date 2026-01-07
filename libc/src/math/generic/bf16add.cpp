//===-- Implementation of bf16add function --------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/math/bf16add.h"
#include "src/__support/FPUtil/bfloat16.h"
#include "src/__support/FPUtil/generic/add_sub.h"
#include "src/__support/common.h"
#include "src/__support/macros/config.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(bfloat16, bf16add, (double x, double y)) {
  return fputil::generic::add<bfloat16>(x, y);
}

} // namespace LIBC_NAMESPACE_DECL
