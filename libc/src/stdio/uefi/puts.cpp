//===-- Implementation of puts for baremetal-------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/stdio/puts.h"
#include "src/__support/UEFI/file.h"
#include "src/__support/macros/config.h"
#include "src/string/strlen.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(int, puts, (const char *__restrict str)) {
  return (int)stdout.write(reinterpret_cast<const void *>(str), strlen(str));
}

} // namespace LIBC_NAMESPACE_DECL
