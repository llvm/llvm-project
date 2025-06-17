//===-- Baremetal Implementation of putc ----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/stdio/putc.h"

#include "src/__support/CPP/string_view.h"
#include "src/__support/macros/config.h"
#include "src/stdio/baremetal/write_utils.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(int, putc, (int c, ::FILE *stream)) {
  char uc = static_cast<char>(c);

  write(stream, cpp::string_view(&uc, 1));

  return 0;
}

} // namespace LIBC_NAMESPACE_DECL
