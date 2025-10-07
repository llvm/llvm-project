//===---------- Linux implementation of the Linux pkey_mprotect function --===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/sys/mman/pkey_set.h"

#include "hdr/errno_macros.h" // For ENOSYS
#include "src/__support/common.h"
#include "src/__support/error_or.h"
#include "src/__support/libc_errno.h"
#include "src/__support/macros/attributes.h"
#include "src/__support/macros/config.h"

#if defined(LIBC_TARGET_ARCH_IS_X86_64)
#include "src/sys/mman/linux/x86_64/pkey_common.h"
#else
#include "src/sys/mman/linux/generic/pkey_common.h"
#endif

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(int, pkey_set, (int pkey, unsigned int access_rights)) {
  ErrorOr<int> ret = LIBC_NAMESPACE::pkey_common::pkey_set(pkey, access_rights);
  if (!ret.has_value()) {
    libc_errno = ret.error();
    return -1;
  }
  return ret.value();
}

} // namespace LIBC_NAMESPACE_DECL
