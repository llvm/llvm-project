//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Implementation header for memfd_create function.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_SYS_MMAN_MEMFD_CREATE_H
#define LLVM_LIBC_SRC_SYS_MMAN_MEMFD_CREATE_H

#include "src/__support/macros/config.h"

namespace LIBC_NAMESPACE_DECL {

int memfd_create(const char *name, unsigned int flags);

} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC_SYS_MMAN_MEMFD_CREATE_H
