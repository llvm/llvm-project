//===-- Implementation of isdigit------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/ctype/isdigit_l.h"
#include "src/ctype/isdigit.h"
#include "src/__support/common.h"

namespace LIBC_NAMESPACE {

LLVM_LIBC_FUNCTION(int, isdigit_l, (int c, locale_t)) {
  return isdigit(c);
}

} // namespace LIBC_NAMESPACE
