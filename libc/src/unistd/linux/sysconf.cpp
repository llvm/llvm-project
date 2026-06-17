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
#include "hdr/unistd_macros.h"
#include "src/__support/OSUtil/linux/auxv.h"
#include "src/__support/OSUtil/linux/sysinfo.h"
#include "src/__support/libc_errno.h"
#include "src/__support/macros/config.h"

namespace LIBC_NAMESPACE_DECL {

namespace { // Anonymous namespace for internal helpers

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

} // anonymous namespace

LLVM_LIBC_FUNCTION(long, sysconf, (int name)) {
  switch (name) {
  case _SC_PAGESIZE:
    return get_page_size();
  case _SC_NPROCESSORS_CONF:
    return get_nprocessors_conf();
  case _SC_NPROCESSORS_ONLN:
    return get_nprocessors_onln();
  case _SC_THREADS:
    return _POSIX_THREADS;
  default:
    // TODO: Complete the rest of the sysconf options.
    libc_errno = EINVAL;
    return -1;
  }
}

} // namespace LIBC_NAMESPACE_DECL
