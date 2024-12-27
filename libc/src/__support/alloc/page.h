//===-- Page allocations ----------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC___SUPPORT_ALLOC_PAGE_H
#define LLVM_LIBC_SRC___SUPPORT_ALLOC_PAGE_H

#include "hdr/types/size_t.h"
#include "src/__support/macros/config.h"

namespace LIBC_NAMESPACE_DECL {

void *page_allocate(size_t n_pages);
void *page_expand(void *ptr, size_t n_pages);
bool page_free(void *ptr, size_t n_pages);

} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC___SUPPORT_ALLOC_BASE_H
