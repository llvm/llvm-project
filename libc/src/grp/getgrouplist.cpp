//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Implementation of getgrouplist.
///
//===----------------------------------------------------------------------===//

#include "src/grp/getgrouplist.h"
#include "hdr/errno_macros.h"
#include "hdr/types/gid_t.h"
#include "src/__support/common.h"
#include "src/__support/libc_errno.h"
#include "src/__support/macros/config.h"
#include "src/__support/macros/null_check.h"
#include "src/grp/grp_utils.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(int, getgrouplist,
                   (const char *user, gid_t group, gid_t *groups,
                    int *ngroups)) {
  LIBC_CRASH_ON_NULLPTR(user);
  LIBC_CRASH_ON_NULLPTR(ngroups);
  if (ngroups && *ngroups > 0)
    LIBC_CRASH_ON_NULLPTR(groups);

  if (!user || !ngroups || *ngroups < 0 || (*ngroups > 0 && !groups)) {
    libc_errno = EINVAL;
    return -1;
  }

  const auto res =
      grp::get_group_list(user, group, groups, static_cast<size_t>(*ngroups));
  if (!res.has_value()) {
    libc_errno = res.error();
    return -1;
  }

  const size_t total = res.value();
  const int requested = *ngroups;
  *ngroups = static_cast<int>(total);

  if (requested < static_cast<int>(total))
    return -1;

  return static_cast<int>(total);
}

} // namespace LIBC_NAMESPACE_DECL
