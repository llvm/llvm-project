//===-- Implementation of the powif function for GPU ----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/math/powif.h"
#include "src/__support/common.h"
#include "src/__support/macros/config.h"

#include "declarations.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(float, powif, (float x, int y)) { return __nv_powif(x, y); }

} // namespace LIBC_NAMESPACE_DECL
