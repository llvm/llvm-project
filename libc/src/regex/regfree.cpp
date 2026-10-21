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
#include "src/__support/macros/null_check.h"
#include "src/__support/regex/regex_internal.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(void, regfree, (regex_t * preg)) {
  LIBC_CRASH_ON_NULLPTR(preg);
  // TODO: Once RegexInternal is managed via cpp::unique_ptr, reset the smart
  // pointer here instead of calling delete manually.
  delete static_cast<regex::RegexInternal *>(preg->__internal);
  preg->__internal = nullptr;
  preg->re_nsub = 0;
}

} // namespace LIBC_NAMESPACE_DECL
