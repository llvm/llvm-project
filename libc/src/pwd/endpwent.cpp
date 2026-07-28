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
#include "src/pwd/getpwent.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(void, endpwent, ()) {
  // endpwent_impl closes the password file. If an error occurs,
  // it returns an Error with an errno value which is set here.
  auto res = endpwent_impl();
  if (!res.has_value())
    libc_errno = res.error();
}

} // namespace LIBC_NAMESPACE_DECL
