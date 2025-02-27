//===-- Implementation of isdigit------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/ctype/isdigit.h"
#include "src/__support/common.h"
#include "src/__support/ctype_utils.h"
#include "src/__support/macros/config.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(int, isdigit, (int c)) {
  return static_cast<int>(internal::isdigit(static_cast<unsigned>(c)));
}

} // namespace LIBC_NAMESPACE_DECL
