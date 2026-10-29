//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Implementation header for close_range.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_UNISTD_CLOSE_RANGE_H
#define LLVM_LIBC_SRC_UNISTD_CLOSE_RANGE_H

#include "src/__support/macros/config.h"

namespace LIBC_NAMESPACE_DECL {

int close_range(unsigned int first, unsigned int last, unsigned int flags);

} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC_UNISTD_CLOSE_RANGE_H
