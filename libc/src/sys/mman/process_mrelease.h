//===-- Implementation header for process_mrelease function -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_SYS_MMAN_PROCESS_MRELEASE_H
#define LLVM_LIBC_SRC_SYS_MMAN_PROCESS_MRELEASE_H

#include "src/__support/macros/config.h"
#include <sys/mman.h> // For size_t and off_t

namespace LIBC_NAMESPACE_DECL {

int process_mrelease(int pidfd, unsigned int flags);

} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC_SYS_MMAN_PROCESS_MRELEASE_H
