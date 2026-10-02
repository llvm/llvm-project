//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Implementation of regfree.
///
//===----------------------------------------------------------------------===//

#include "src/regex/regfree.h"

#include "src/__support/CPP/new.h"
#include "src/__support/common.h"
#include "src/__support/macros/config.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(void, regfree, (regex_t * preg)) {
  if (preg->__internal) {
    char *ptr = static_cast<char *>(preg->__internal);
    delete[] ptr;
    preg->__internal = nullptr;
  }
  preg->re_nsub = 0;
}

} // namespace LIBC_NAMESPACE_DECL
