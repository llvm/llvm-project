//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Linux implementation of makedev.
///
//===----------------------------------------------------------------------===//

#include "src/sys/sysmacros/makedev.h"
#include "hdr/types/dev_t.h"
#include "src/__support/common.h"
#include "src/__support/macros/config.h"

namespace LIBC_NAMESPACE_DECL {

static_assert(sizeof(dev_t) == 8, "dev_t must be 64-bit");

// dev_t is encoded as MMMM Mmmm mmmM MMmm, where each M represents 4 bytes
// of the major number and m represents 4 bytes of the minor number. We don't
// support older systems with smaller dev_t types.

LLVM_LIBC_FUNCTION(dev_t, makedev, (unsigned int major, unsigned int minor)) {
  return (static_cast<dev_t>(major & 0x00000fff) << 8) |
         (static_cast<dev_t>(major & 0xfffff000) << 32) |
         (static_cast<dev_t>(minor & 0x000000ff) << 0) |
         (static_cast<dev_t>(minor & 0xffffff00) << 12);
}

} // namespace LIBC_NAMESPACE_DECL
