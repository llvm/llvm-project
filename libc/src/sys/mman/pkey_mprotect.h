//===-- Implementation header for pkey_mprotect function --------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_SYS_MMAN_PKEY_MPROTECT_H
#define LLVM_LIBC_SRC_SYS_MMAN_PKEY_MPROTECT_H

#include "hdr/types/size_t.h"
#include "src/__support/macros/config.h"

namespace LIBC_NAMESPACE_DECL {

int pkey_mprotect(void *addr, size_t len, int prot, int pkey);

} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC_SYS_MMAN_PKEY_MPROTECT_H
