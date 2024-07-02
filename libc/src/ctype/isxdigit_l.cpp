//===-- Implementation of isxdigit_l---------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/ctype/isxdigit_l.h"
#include "src/__support/ctype_utils.h"
#include "src/ctype/isxdigit.h"

#include "src/__support/common.h"

namespace LIBC_NAMESPACE {

// TODO: Currently restricted to default locale.
// These should be extended using locale information.
LLVM_LIBC_FUNCTION(int, isxdigit_l, (int c, locale_t)) { return isxdigit(c); }

} // namespace LIBC_NAMESPACE
