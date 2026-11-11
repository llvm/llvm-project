//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Implementation of endgrent.
///
//===----------------------------------------------------------------------===//

#include "src/grp/endgrent.h"
#include "src/__support/common.h"
#include "src/__support/libc_errno.h"
#include "src/grp/grp_utils.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(void, endgrent, ()) {
  auto res = grp::close();
  if (!res.has_value())
    libc_errno = res.error();
}

} // namespace LIBC_NAMESPACE_DECL
