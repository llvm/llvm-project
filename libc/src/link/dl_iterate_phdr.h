//===-- Implementation header for dl_iterate_phdr ---------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_LINK_DL_ITERATE_PHDR_H
#define LLVM_LIBC_SRC_LINK_DL_ITERATE_PHDR_H

#include "include/llvm-libc-types/__dl_iterate_phdr_callback_t.h"
#include "src/__support/macros/config.h"

namespace LIBC_NAMESPACE_DECL {

int dl_iterate_phdr(__dl_iterate_phdr_callback_t callback, void *data);

} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC_STRING_MEMCHR_H
