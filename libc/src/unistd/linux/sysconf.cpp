//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Linux implementation of sysconf
///
//===----------------------------------------------------------------------===//

#include "src/unistd/sysconf.h"

#include "src/__support/common.h"

#include "hdr/sys_auxv_macros.h"
#include "hdr/sys_resource_macros.h"
#include "hdr/types/struct_rlimit.h"
#include "hdr/unistd_macros.h"
#include "src/__support/OSUtil/linux/auxv.h"
#include "src/__support/OSUtil/linux/syscall_wrappers/prlimit.h"
#include "src/__support/OSUtil/linux/syscall_wrappers/sysinfo.h"
#include "src/__support/OSUtil/linux/sysinfo.h"
#include "src/__support/libc_errno.h"
#include "src/__support/macros/config.h"
#include <linux/limits.h>
#include <linux/sysinfo.h>

// In overlay mode, system headers (like glibc's <bits/local_lim.h>) may
// explicitly undefine ARG_MAX to indicate it is dynamic. We define a fallback
// here using the standard Linux kernel minimum floor of 128KB.
#ifndef ARG_MAX
#define ARG_MAX 131072
#endif

namespace LIBC_NAMESPACE_DECL {

namespace { // Anonymous namespace for internal helpers

// Fallback value for ARG_MAX when RLIMIT_STACK is RLIM_INFINITY (unlimited).
// When the stack is unlimited, the kernel caps the argument limit at 3/4 of the
// default stack limit (_STK_LIM = 8MB), which yields 6MB.
constexpr long DEFAULT_STACK_LIMIT = 8 * 1024 * 1024;          // 8MB
constexpr long ARG_MAX_FALLBACK = DEFAULT_STACK_LIMIT / 4 * 3; // 6MB

// We define a local structure for prlimit64 to avoid type mismatches
// and stack corruption on 32-bit systems when in overlay mode.
struct rlimit64 {
  uint64_t rlim_cur;
  uint64_t rlim_max;
};

long get_arg_max() {
  struct rlimit64 limits;
  ErrorOr<int> ret = linux_syscalls::prlimit(
      0, RLIMIT_STACK, nullptr, reinterpret_cast<struct rlimit *>(&limits));
  if (!ret) {
    libc_errno = -ret.error();
    return -1;
  }
  if (limits.rlim_cur == ~0ULL)
    return ARG_MAX_FALLBACK;

  long val = static_cast<long>(limits.rlim_cur / 4);
  return val > ARG_MAX ? val : ARG_MAX;
}

long get_open_max() {
  struct rlimit64 limits;
  ErrorOr<int> ret = linux_syscalls::prlimit(
      0, RLIMIT_NOFILE, nullptr, reinterpret_cast<struct rlimit *>(&limits));
  if (!ret) {
    libc_errno = -ret.error();
    return -1;
  }
  if (limits.rlim_cur == ~0ULL)
    return -1;
  return static_cast<long>(limits.rlim_cur);
}

long get_page_size() {
  cpp::optional<unsigned long> page_size = auxv::get(AT_PAGESZ);
  if (page_size)
    return static_cast<long>(*page_size);
  libc_errno = EINVAL;
  return -1;
}

long get_nprocessors_conf() {
  return static_cast<long>(
      sysinfo::parse_nproc_with_fallback_from(sysinfo::POSSIBLE_NPROC_PATH));
}

long get_nprocessors_onln() {
  return static_cast<long>(
      sysinfo::parse_nproc_with_fallback_from(sysinfo::ONLINE_NPROC_PATH));
}

long get_phys_pages() {
  struct ::sysinfo info;
  ErrorOr<int> ret = linux_syscalls::sysinfo(&info);
  if (!ret) {
    libc_errno = -ret.error();
    return -1;
  }
  cpp::optional<unsigned long> page_size = auxv::get(AT_PAGESZ);
  if (!page_size) {
    libc_errno = ENOSYS;
    return -1;
  }
  unsigned long ps = *page_size;
  unsigned long mem_unit = info.mem_unit;
  unsigned long num = info.totalram;
  if (mem_unit >= ps) {
    num *= (mem_unit / ps);
  } else {
    num /= (ps / mem_unit);
  }
  return static_cast<long>(num);
}

} // anonymous namespace

LLVM_LIBC_FUNCTION(long, sysconf, (int name)) {
  switch (name) {
  case _SC_ARG_MAX:
    return get_arg_max();
  case _SC_PAGESIZE:
    return get_page_size();
  case _SC_NPROCESSORS_CONF:
    return get_nprocessors_conf();
  case _SC_NPROCESSORS_ONLN:
    return get_nprocessors_onln();
  case _SC_THREADS:
    return _POSIX_THREADS;
  case _SC_OPEN_MAX:
    return get_open_max();
  case _SC_PHYS_PAGES:
    return get_phys_pages();
  default:
    // TODO: Complete the rest of the sysconf options.
    libc_errno = EINVAL;
    return -1;
  }
}

} // namespace LIBC_NAMESPACE_DECL
