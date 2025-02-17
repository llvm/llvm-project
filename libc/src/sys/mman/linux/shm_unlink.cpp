//===---------- Linux implementation of the shm_unlink function -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/sys/mman/shm_unlink.h"
#include "src/__support/macros/config.h"
#include "src/sys/mman/linux/shm_common.h"
#include "src/unistd/unlink.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(int, shm_unlink, (const char *name)) {
  using namespace shm_common;
  if (cpp::optional<SHMPath> buffer = translate_name(name))
    return unlink(buffer->data());
  return -1;
}

} // namespace LIBC_NAMESPACE_DECL
