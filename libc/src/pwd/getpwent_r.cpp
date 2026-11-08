//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Implementation of getpwent_r.
///
//===----------------------------------------------------------------------===//

#include "src/pwd/getpwent_r.h"
#include "hdr/errno_macros.h"
#include "hdr/types/size_t.h"
#include "hdr/types/struct_passwd.h"
#include "src/__support/CPP/span.h"
#include "src/__support/common.h"
#include "src/__support/macros/config.h"
#include "src/__support/macros/null_check.h"
#include "src/pwd/pwd_utils.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(int, getpwent_r,
                   (struct passwd * pwbuf, char *buf, size_t buflen,
                    struct passwd **pwbufp)) {
  LIBC_CRASH_ON_NULLPTR(pwbuf);
  LIBC_CRASH_ON_NULLPTR(buf);
  LIBC_CRASH_ON_NULLPTR(pwbufp);

  *pwbufp = nullptr;

  const auto res = pwd::read_next(pwbuf, cpp::span<char>(buf, buflen));
  if (!res.has_value())
    return res.error();

  if (!res.value())
    return ENOENT;

  *pwbufp = pwbuf;
  return 0;
}

} // namespace LIBC_NAMESPACE_DECL
