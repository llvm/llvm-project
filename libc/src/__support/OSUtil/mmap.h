//===----------- Implementation header of the mmap function -----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC___SUPPORT_OSUTIL_MMAP_H
#define LLVM_LIBC_SRC___SUPPORT_OSUTIL_MMAP_H

#include "src/__support/error_or.h"
#include "src/__support/macros/config.h"
#include <sys/mman.h> // For size_t and off_t

namespace LIBC_NAMESPACE_DECL {
namespace internal {

ErrorOr<void *> mmap(void *addr, size_t size, int prot, int flags, int fd,
                     off_t offset);

} // namespace internal
} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC___SUPPORT_OSUTIL_MMAP_H
