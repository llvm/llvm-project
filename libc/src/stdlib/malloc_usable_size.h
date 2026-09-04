//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Implementation header for malloc_usable_size.
///
//===----------------------------------------------------------------------===//

#include "src/__support/macros/config.h"
#include <stddef.h>

#ifndef LLVM_LIBC_SRC_STDLIB_MALLOC_USABLE_SIZE_H
#define LLVM_LIBC_SRC_STDLIB_MALLOC_USABLE_SIZE_H

namespace LIBC_NAMESPACE_DECL {

size_t malloc_usable_size(void *ptr);

} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC_STDLIB_MALLOC_USABLE_SIZE_H
