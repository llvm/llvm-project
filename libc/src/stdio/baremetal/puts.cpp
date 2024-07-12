//===-- Implementation of puts for baremetal-------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/stdio/puts.h"
#include "src/__support/CPP/string_view.h"
#include "src/__support/OSUtil/io.h"
#include "src/__support/macros/config.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(int, puts, (const char *__restrict str)) {
  cpp::string_view str_view(str);

  // TODO: Can we combine these to avoid needing two writes?
  write_to_stderr(str_view);
  write_to_stderr("\n");

  return 0;
}

} // namespace LIBC_NAMESPACE_DECL
