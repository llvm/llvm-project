//===-- Implementation of atoi --------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/stdlib/atoi.h"
#include "src/__support/macros/config.h"
#include "src/stdlib/str_to_util.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(int, atoi, (const char *str)) {
  // This is done because the standard specifies that atoi is identical to
  // (int)(strtol).
  return static_cast<int>(
      internal::str_to_helper<long, char>(str, nullptr, 10));
}

} // namespace LIBC_NAMESPACE_DECL
