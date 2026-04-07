//===-- Internal header for baremetal abort -------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_STDLIB_BAREMETAL_ABORT_UTILS_H
#define LLVM_LIBC_SRC_STDLIB_BAREMETAL_ABORT_UTILS_H

#include "src/__support/common.h"
#include "src/__support/macros/config.h"

namespace LIBC_NAMESPACE_DECL {

namespace abort_utils {
[[noreturn]] LIBC_INLINE void abort() { __builtin_trap(); }
} // namespace abort_utils

} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC_STDLIB_BAREMETAL_ABORT_UTILS_H
