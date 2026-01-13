//===---------- Generic stub implementations for pkey functionality. ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_SYS_MMAN_LINUX_GENERIC_PKEY_COMMON_H_
#define LLVM_SYS_MMAN_LINUX_GENERIC_PKEY_COMMON_H_

#include "hdr/errno_macros.h" // For ENOSYS
#include "src/__support/common.h"
#include "src/__support/error_or.h"

namespace LIBC_NAMESPACE_DECL {
namespace pkey_common {

LIBC_INLINE ErrorOr<int> pkey_get([[maybe_unused]] int pkey) {
  return Error(ENOSYS);
}

LIBC_INLINE ErrorOr<int> pkey_set([[maybe_unused]] int pkey,
                                  [[maybe_unused]] unsigned int access_rights) {
  return Error(ENOSYS);
}

} // namespace pkey_common
} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_SYS_MMAN_LINUX_GENERIC_PKEY_COMMON_H_
