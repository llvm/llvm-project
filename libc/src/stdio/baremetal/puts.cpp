//===-- Implementation of puts for baremetal-------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/stdio/puts.h"

#include "hdr/stdio_macros.h" // for EOF
#include "src/__support/common.h"
#include "src/__support/libc_errno.h"
#include "src/__support/macros/config.h"
#include "src/stdio/baremetal/file_internal.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(int, puts, (const char *__restrict str)) {
  cpp::string_view str_view(str);
  auto result = write_internal(str_view.data(), str_view.size(), stdout);
  if (result.has_error())
    libc_errno = result.error;
  size_t written = result.value;
  if (written != str_view.size()) {
    // The stream should be in an error state in this case.
    return EOF;
  }
  result = write_internal("\n", 1, stdout);
  if (result.has_error())
    libc_errno = result.error;
  written = result.value;
  if (written != 1) {
    // The stream should be in an error state in this case.
    return EOF;
  }
  return 0;
}

} // namespace LIBC_NAMESPACE_DECL
