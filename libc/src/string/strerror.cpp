//===-- Implementation of strerror ----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/string/strerror.h"
#include "src/__support/CPP/span.h"
#include "src/__support/common.h"
#include "src/__support/error_to_string.h"

namespace __llvm_libc {

LLVM_LIBC_FUNCTION(char *, strerror, (int err_num)) {
  return const_cast<char *>(get_error_string(err_num).data());
}

} // namespace __llvm_libc
