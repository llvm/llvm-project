//===-- Implementation of the cosf function for GPU -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/math/cosf.h"
#include "src/__support/common.h"

#include "common.h"

namespace __llvm_libc {

#if defined(__CLANG_GPU_APPROX_TRANSCENDENTALS__)
namespace fast {
  LLVM_LIBC_FUNCTION(float, cosf, (float x)) { return __llvm_libc::internal::fast::cosf(x); }
}
#else
LLVM_LIBC_FUNCTION(float, cosf, (float x)) { return internal::cosf(x); }
#endif

} // namespace __llvm_libc
