//===-- Implementation of strcspn -----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/string/strcspn.h"

#include "src/__support/common.h"
#include "src/string/string_utils.h"

namespace __llvm_libc {

size_t LLVM_LIBC_ENTRYPOINT(strcspn)(const char *src, const char *segment) {
  return internal::complementary_span(src, segment);
}

} // namespace __llvm_libc
