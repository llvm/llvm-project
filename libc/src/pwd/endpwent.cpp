//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Implementation of endpwent.
///
//===----------------------------------------------------------------------===//

#include "src/pwd/endpwent.h"
#include "src/__support/common.h"
#include "src/__support/libc_errno.h"
#include "src/pwd/pwd_utils.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(void, endpwent, ()) {
  auto res = pwd::close();
  if (!res.has_value())
    libc_errno = res.error();
}

} // namespace LIBC_NAMESPACE_DECL
