//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Implementation header for backtrace
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_EXECINFO_BACKTRACE_H
#define LLVM_LIBC_SRC_EXECINFO_BACKTRACE_H

#include "src/__support/macros/config.h"

namespace LIBC_NAMESPACE_DECL {

int backtrace(void **buffer, int size);

} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC_EXECINFO_BACKTRACE_H
