//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Linux implementation of waitid.
///
//===----------------------------------------------------------------------===//

#include "src/sys/wait/waitid.h"

#include "hdr/types/id_t.h"
#include "hdr/types/idtype_t.h"
#include "hdr/types/siginfo_t.h"
#include "src/__support/OSUtil/linux/syscall_wrappers/waitid.h"
#include "src/__support/common.h"
#include "src/__support/libc_errno.h"
#include "src/__support/macros/config.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(int, waitid,
                   (idtype_t idtype, id_t id, siginfo_t *infop, int options)) {
  auto result = linux_syscalls::waitid(idtype, id, infop, options, nullptr);
  if (!result.has_value()) {
    libc_errno = result.error();
    return -1;
  }
  return 0;
}

} // namespace LIBC_NAMESPACE_DECL
