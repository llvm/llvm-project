//===-- Implementation of the expm1f function for GPU ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/math/expm1f.h"
#include "src/__support/common.h"

#include "declarations.h"

namespace LIBC_NAMESPACE {

LLVM_LIBC_FUNCTION(float, expm1f, (float x)) { return __ocml_expm1_f32(x); }

} // namespace LIBC_NAMESPACE
