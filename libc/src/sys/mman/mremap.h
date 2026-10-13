//===-- Implementation header for mremap function -------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_SYS_MMAN_MREMAP_H
#define LLVM_LIBC_SRC_SYS_MMAN_MREMAP_H

#include "hdr/types/size_t.h"
#include "src/__support/macros/config.h"

namespace LIBC_NAMESPACE_DECL {

void *mremap(void *old_address, size_t old_size, size_t new_size, int flags,
             ... /* void *new_address */);

} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC_SYS_MMAN_MREMAP_H
