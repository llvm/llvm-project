//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Implementation header for statx.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_SYS_STAT_STATX_H
#define LLVM_LIBC_SRC_SYS_STAT_STATX_H

#include "hdr/types/struct_statx.h"
#include "src/__support/macros/config.h"

namespace LIBC_NAMESPACE_DECL {

int statx(int dirfd, const char *__restrict path, int flags, unsigned int mask,
          struct statx *__restrict statxbuf);

} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC_SYS_STAT_STATX_H
