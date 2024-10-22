//===-- Implementation of crealf16 function -------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/complex/crealf16.h"
#if defined(LIBC_TYPES_HAS_CFLOAT16)

#include "src/__support/FPUtil/NearestIntegerOperations.h"
#include "src/__support/common.h"
#include "src/__support/macros/config.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(float16, crealf16, (cfloat16 x)) {
    float16 *xCmplxPtr = reinterpret_cast<float16 *>(&x);
    return xCmplxPtr[0];
}

} // namespace LIBC_NAMESPACE_DECL

#endif // LIBC_TYPES_HAS_CFLOAT16
