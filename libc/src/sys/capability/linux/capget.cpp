//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Linux implementation of capget.
///
//===----------------------------------------------------------------------===//

#include "src/sys/capability/capget.h"

#include "hdr/types/cap_user_data_t.h"
#include "hdr/types/cap_user_header_t.h"
#include "src/__support/OSUtil/linux/syscall_wrappers/capget.h"
#include "src/__support/common.h"
#include "src/__support/error_or.h"
#include "src/__support/libc_errno.h"
#include "src/__support/macros/config.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(int, capget,
                   (cap_user_header_t hdrp, cap_user_data_t datap)) {
  auto result = linux_syscalls::capget(hdrp, datap);
  if (!result.has_value()) {
    libc_errno = result.error();
    return -1;
  }
  return result.value();
}

} // namespace LIBC_NAMESPACE_DECL
