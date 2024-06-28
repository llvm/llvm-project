//===-- Baremetal Implementation of putchar -------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/stdio/putchar.h"
#include "src/__support/CPP/string_view.h"
#include "src/__support/OSUtil/io.h"

namespace LIBC_NAMESPACE {

LLVM_LIBC_FUNCTION(int, putchar, (int c)) {
  char uc = static_cast<char>(c);

  write_to_stderr(cpp::string_view(&uc, 1));

  return 0;
}

} // namespace LIBC_NAMESPACE
