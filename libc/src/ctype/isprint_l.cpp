//===-- Implementation of isprint -----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/ctype/isprint_l.h"

#include "src/__support/common.h"
#include "src/__support/macros/config.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(int, isprint_l, (int c, locale_t)) {
  const unsigned ch = static_cast<unsigned>(c);
  return static_cast<int>((ch - ' ') < 95);
}

} // namespace LIBC_NAMESPACE_DECL
