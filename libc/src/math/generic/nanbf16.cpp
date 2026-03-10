//===-- Implementation of nanbf16 function --------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/math/nanbf16.h"
#include "src/__support/FPUtil/bfloat16.h"
#include "src/__support/common.h"
#include "src/__support/libc_errno.h"
#include "src/__support/macros/config.h"
#include "src/__support/str_to_float.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(bfloat16, nanbf16, (const char *arg)) {
  auto result = internal::strtonan<bfloat16>(arg);
  if (result.has_error())
    libc_errno = result.error;
  return result.value;
}

} // namespace LIBC_NAMESPACE_DECL
