//===-- Implementation header for getppid -----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_UNISTD_GETPPID_H
#define LLVM_LIBC_SRC_UNISTD_GETPPID_H

#include "src/__support/macros/config.h"
#include <unistd.h>

namespace LIBC_NAMESPACE_DECL {

pid_t getppid();

} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC_UNISTD_GETPPID_H
