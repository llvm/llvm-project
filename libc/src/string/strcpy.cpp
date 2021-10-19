//===-- Implementation of strcpy ------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/string/strcpy.h"
#include "src/string/memcpy.h"
#include "src/string/string_utils.h"

#include "src/__support/common.h"
#include "src/__support/sanitizer.h"

namespace __llvm_libc {

LLVM_LIBC_FUNCTION(char *, strcpy,
                   (char *__restrict dest, const char *__restrict src)) {
  size_t size = internal::string_length(src) + 1;
  char *result = reinterpret_cast<char *>(__llvm_libc::memcpy(dest, src, size));

  // In many libc uses, we do not want memcpy to be instrumented. Hence,
  // we mark the destination as initialized.
  //
  // We do not want memcpy to be instrumented because compilers can potentially
  // generate calls to memcpy. If the sanitizer business logic ends up with a
  // compiler generated call to memcpy which is instrumented, then it will
  // break the sanitizers.
  SANITIZER_MEMORY_INITIALIZED(result, size);

  return result;
}

} // namespace __llvm_libc
