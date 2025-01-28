//===-- Implementation header for syscall (va compat) -----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_UNISTD_SYSCALL_COMPAT_H
#define LLVM_LIBC_SRC_UNISTD_SYSCALL_COMPAT_H

#include "src/__support/macros/config.h"

namespace LIBC_NAMESPACE_DECL {
#pragma push_macro("syscall")
#undef syscall
long syscall(long number, ...);
#pragma pop_macro("syscall")
} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC_UNISTD_SYSCALL_COMPAT_H
