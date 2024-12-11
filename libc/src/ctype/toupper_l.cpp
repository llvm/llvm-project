//===-- Implementation of toupper_l ---------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/ctype/toupper_l.h"
#include "src/__support/ctype_utils.h"

#include "src/__support/common.h"
#include "src/__support/macros/config.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(int, toupper_l, (int c, locale_t)) {
  if (internal::islower(c))
    return c - ('a' - 'A');
  return c;
}

} // namespace LIBC_NAMESPACE_DECL
