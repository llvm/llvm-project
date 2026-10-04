//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Implementation of getgrnam.
///
//===----------------------------------------------------------------------===//

#include "src/grp/getgrnam.h"
#include "hdr/types/struct_group.h"
#include "src/__support/CPP/string_view.h"
#include "src/__support/common.h"
#include "src/__support/libc_errno.h"
#include "src/__support/macros/config.h"
#include "src/__support/macros/null_check.h"
#include "src/grp/grp_utils.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(struct group *, getgrnam, (const char *name)) {
  LIBC_CRASH_ON_NULLPTR(name);

  const auto res = grp::find_by_name(name);
  if (!res.has_value()) {
    libc_errno = res.error();
    return nullptr;
  }
  return res.value();
}

} // namespace LIBC_NAMESPACE_DECL
