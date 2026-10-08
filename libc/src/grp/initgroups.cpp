//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Implementation of initgroups.
///
//===----------------------------------------------------------------------===//

#include "src/grp/initgroups.h"
#include "hdr/types/gid_t.h"
#include "src/__support/common.h"
#include "src/__support/libc_errno.h"
#include "src/__support/macros/config.h"
#include "src/__support/macros/null_check.h"
#include "src/grp/grp_utils.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(int, initgroups, (const char *user, gid_t group)) {
  LIBC_CRASH_ON_NULLPTR(user);

  const auto res = grp::init_groups(user, group);
  if (!res.has_value()) {
    libc_errno = res.error();
    return -1;
  }
  return 0;
}

} // namespace LIBC_NAMESPACE_DECL
