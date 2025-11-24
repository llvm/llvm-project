//===---------- Linux implementation of the Linux pkey_mprotect function --===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/sys/mman/pkey_mprotect.h"

#include "hdr/errno_macros.h" // For ENOSYS
#include "hdr/types/size_t.h"
#include "src/__support/OSUtil/syscall.h" // For internal syscall function.
#include "src/__support/common.h"
#include "src/__support/error_or.h"
#include "src/__support/libc_errno.h"
#include "src/__support/macros/config.h"
#include "src/sys/mman/linux/mprotect_common.h"

#include <sys/syscall.h> // For syscall numbers.

namespace LIBC_NAMESPACE_DECL {
namespace internal {

LIBC_INLINE ErrorOr<int> pkey_mprotect_impl(void *addr, size_t len, int prot,
                                            int pkey) {
  // Fall back to mprotect if pkey is -1
  // to maintain compatibility with kernel versions that don't support pkey.
  if (pkey == -1) {
    return LIBC_NAMESPACE::mprotect_common::mprotect_impl(addr, len, prot);
  }

#if !defined(SYS_pkey_mprotect)
  return Error(ENOSYS);
#else
  int ret = LIBC_NAMESPACE::syscall_impl<int>(SYS_pkey_mprotect, addr, len,
                                              prot, pkey);
  if (ret < 0) {
    return Error(-ret);
  }
  return 0;
#endif
}

} // namespace internal

LLVM_LIBC_FUNCTION(int, pkey_mprotect,
                   (void *addr, size_t len, int prot, int pkey)) {
  ErrorOr<int> ret =
      LIBC_NAMESPACE::internal::pkey_mprotect_impl(addr, len, prot, pkey);
  if (!ret.has_value()) {
    libc_errno = ret.error();
    return -1;
  }
  return ret.value();
}

} // namespace LIBC_NAMESPACE_DECL
