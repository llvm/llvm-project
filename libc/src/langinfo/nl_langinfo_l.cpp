//===-- Implementation of nl_langinfo_l -----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/langinfo/nl_langinfo_l.h"

#include "src/__support/common.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(char *, nl_langinfo_l, (nl_item, locale_t locale)) {
  static char EMPTY[] = "";
  return EMPTY;
}

} // namespace LIBC_NAMESPACE_DECL
