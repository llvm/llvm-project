//===---------- Linux implementation of the shm_unlink function -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/sys/mman/shm_unlink.h"
#include "src/sys/mman/linux/shm_common.h"
#include "src/unistd/unlink.h"

namespace LIBC_NAMESPACE {

LLVM_LIBC_FUNCTION(int, shm_unlink, (const char *name)) {
  cpp::optional<SHMPath> buffer = get_shm_name(name);
  if (!buffer.has_value())
    return -1;
  return unlink(buffer->data());
}

} // namespace LIBC_NAMESPACE
