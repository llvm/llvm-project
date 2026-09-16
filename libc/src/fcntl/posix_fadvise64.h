//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Implementation header for posix_fadvise64.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_FCNTL_POSIX_FADVISE64_H
#define LLVM_LIBC_SRC_FCNTL_POSIX_FADVISE64_H

#include "hdr/types/off64_t.h"
#include "src/__support/macros/config.h"

namespace LIBC_NAMESPACE_DECL {

int posix_fadvise64(int fd, off64_t offset, off64_t len, int advice);

} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC_FCNTL_POSIX_FADVISE64_H
