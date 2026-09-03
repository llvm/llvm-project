//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Linux implementation of posix_fadvise.
///
//===----------------------------------------------------------------------===//

#include "src/fcntl/posix_fadvise.h"

#include "hdr/stdint_proxy.h"
#include "hdr/types/off_t.h"
#include "src/__support/CPP/bit.h"
#include "src/__support/OSUtil/syscall.h"
#include "src/__support/common.h"
#include "src/__support/macros/config.h"

#include <sys/syscall.h>

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(int, posix_fadvise,
                   (int fd, off_t offset, off_t len, int advice)) {
  int ret;
  if constexpr (sizeof(long) == sizeof(uint32_t) &&
                sizeof(off_t) == sizeof(uint64_t)) {
    uint64_t offset_bits = cpp::bit_cast<uint64_t>(offset);
    long offset_low = static_cast<long>(offset_bits & UINT32_MAX);
    long offset_high = static_cast<long>(offset_bits >> 32);

#if defined(SYS_fadvise64_64) || defined(SYS_arm_fadvise64_64)
    uint64_t len_bits = cpp::bit_cast<uint64_t>(len);
    long len_low = static_cast<long>(len_bits & UINT32_MAX);
    long len_high = static_cast<long>(len_bits >> 32);
#endif

#if defined(SYS_fadvise64_64)
    ret = LIBC_NAMESPACE::syscall_impl<int>(SYS_fadvise64_64, fd, offset_low,
                                            offset_high, len_low, len_high,
                                            advice);
#elif defined(SYS_arm_fadvise64_64)
    ret = LIBC_NAMESPACE::syscall_impl<int>(SYS_arm_fadvise64_64, fd, advice,
                                            offset_low, offset_high, len_low,
                                            len_high);
#elif defined(SYS_fadvise64)
    ret = LIBC_NAMESPACE::syscall_impl<int>(SYS_fadvise64, fd, offset_low,
                                            offset_high,
                                            static_cast<size_t>(len), advice);
#else
#error "fadvise64 syscall not available."
#endif
  } else {
#if defined(SYS_fadvise64)
    ret = LIBC_NAMESPACE::syscall_impl<int>(SYS_fadvise64, fd, offset, len,
                                            advice);
#elif defined(SYS_fadvise64_64)
    ret = LIBC_NAMESPACE::syscall_impl<int>(SYS_fadvise64_64, fd, offset, len,
                                            advice);
#else
#error "fadvise64 syscall not available."
#endif
  }

  if (ret < 0)
    return -ret;
  return 0;
}

} // namespace LIBC_NAMESPACE_DECL
