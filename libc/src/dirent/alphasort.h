//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Declaration of the POSIX alphasort function.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_DIRENT_ALPHASORT_H
#define LLVM_LIBC_SRC_DIRENT_ALPHASORT_H

#include "hdr/types/struct_dirent.h"
#include "src/__support/macros/config.h"

namespace LIBC_NAMESPACE_DECL {

int alphasort(const struct dirent **a, const struct dirent **b);

} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC_DIRENT_ALPHASORT_H
