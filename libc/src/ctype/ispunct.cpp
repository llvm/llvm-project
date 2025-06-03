//===-- Implementation of ispunct------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/ctype/ispunct.h"

#include "src/__support/common.h"
#include "src/__support/ctype_utils.h"
#include "src/__support/macros/config.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(int, ispunct, (int c)) {
  const unsigned ch = static_cast<unsigned>(c);
  return static_cast<int>(!internal::isalnum(ch) && internal::isgraph(ch));
}

} // namespace LIBC_NAMESPACE_DECL
