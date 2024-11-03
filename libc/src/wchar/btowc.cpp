//===-- Implementation of btowc -------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/wchar/btowc.h"
#include "src/__support/common.h"
#include "src/__support/wctype_utils.h"

#include <stdio.h> // for EOF.

namespace __llvm_libc {

LLVM_LIBC_FUNCTION(int, btowc, (wint_t c)) {
  auto result = internal::btowc(c);
  if (result.has_value()) {
    return result.value();
  } else {
    return WEOF;
  }
}

} // namespace __llvm_libc
