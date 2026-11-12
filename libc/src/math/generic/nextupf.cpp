//===-- Implementation of nextupf function --------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/math/nextupf.h"
#include "src/__support/math/nextupf.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(float, nextupf, (float x)) { return math::nextupf(x); }

} // namespace LIBC_NAMESPACE_DECL
