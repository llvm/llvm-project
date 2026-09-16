//===-- Implementation header for tfind -------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_SEARCH_TFIND_H
#define LLVM_LIBC_SRC_SEARCH_TFIND_H

#include "hdr/types/posix_tnode.h"
#include "src/__support/macros/config.h"

namespace LIBC_NAMESPACE_DECL {
__llvm_libc_tnode *tfind(const void *key, __llvm_libc_tnode *const *rootp,
                         int (*compar)(const void *, const void *));
} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC_SEARCH_TFIND_H
