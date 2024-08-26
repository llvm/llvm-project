//===-- Implementation of isxdigit ----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/ctype/isxdigit_l.h"
#include "src/__support/ctype_utils.h"

#include "src/__support/common.h"
#include "src/__support/macros/config.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(int, isxdigit_l, (int c, locale_t)) {
  const unsigned ch = static_cast<unsigned>(c);
  return static_cast<int>(internal::isdigit(ch) || (ch | 32) - 'a' < 6);
}

} // namespace LIBC_NAMESPACE_DECL
