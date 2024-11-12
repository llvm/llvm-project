//===-- Implementation of toupper------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/ctype/toupper.h"
#include "src/__support/ctype_utils.h"

#include "src/__support/common.h"
#include "src/__support/macros/config.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(int, toupper, (int c)) {
  if (internal::islower(c))
    return c - ('a' - 'A');
  return c;
}

} // namespace LIBC_NAMESPACE_DECL
