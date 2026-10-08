//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Implementation of getpwuid_r.
///
//===----------------------------------------------------------------------===//

#include "src/pwd/getpwuid_r.h"
#include "hdr/types/size_t.h"
#include "hdr/types/struct_passwd.h"
#include "hdr/types/uid_t.h"
#include "src/__support/CPP/span.h"
#include "src/__support/common.h"
#include "src/__support/macros/null_check.h"
#include "src/pwd/pwd_utils.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(int, getpwuid_r,
                   (uid_t uid, struct passwd *pwd, char *buffer, size_t bufsize,
                    struct passwd **result)) {
  LIBC_CRASH_ON_NULLPTR(pwd);
  LIBC_CRASH_ON_NULLPTR(buffer);
  LIBC_CRASH_ON_NULLPTR(result);

  *result = nullptr;

  auto res = pwd::find_by_uid(uid, pwd, cpp::span<char>(buffer, bufsize));
  if (!res.has_value())
    return res.error();

  *result = res.value() ? pwd : nullptr;
  return 0;
}

} // namespace LIBC_NAMESPACE_DECL
