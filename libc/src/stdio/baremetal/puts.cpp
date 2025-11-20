//===-- Implementation of puts for baremetal-------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/stdio/puts.h"
#include "src/__support/common.h"
#include "src/__support/macros/config.h"
#include "src/stdio/baremetal/fputs_internal.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(int, puts, (const char *__restrict str)) {
  return fputs_internal(str, stdout);
}

} // namespace LIBC_NAMESPACE_DECL
