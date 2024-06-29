//===-- Implementation header for insque ------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_SEARCH_INSQUE_H
#define LLVM_LIBC_SRC_SEARCH_INSQUE_H

#include <search.h>

namespace LIBC_NAMESPACE {

void insque(void *elem, void *prev);

} // namespace LIBC_NAMESPACE

#endif // LLVM_LIBC_SRC_SEARCH_INSQUE_H
