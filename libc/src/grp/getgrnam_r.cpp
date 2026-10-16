//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Implementation of getgrnam_r.
///
//===----------------------------------------------------------------------===//

#include "src/grp/getgrnam_r.h"
#include "hdr/types/size_t.h"
#include "hdr/types/struct_group.h"
#include "src/__support/CPP/span.h"
#include "src/__support/CPP/string_view.h"
#include "src/__support/common.h"
#include "src/__support/macros/null_check.h"
#include "src/grp/grp_utils.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(int, getgrnam_r,
                   (const char *name, struct group *grp, char *buffer,
                    size_t bufsize, struct group **result)) {
  LIBC_CRASH_ON_NULLPTR(name);
  LIBC_CRASH_ON_NULLPTR(grp);
  LIBC_CRASH_ON_NULLPTR(buffer);
  LIBC_CRASH_ON_NULLPTR(result);

  *result = nullptr;

  const auto res =
      grp::find_by_name(name, grp, cpp::span<char>(buffer, bufsize));
  if (!res.has_value())
    return res.error();

  *result = res.value() ? grp : nullptr;
  return 0;
}

} // namespace LIBC_NAMESPACE_DECL
