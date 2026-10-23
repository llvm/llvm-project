//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Implementation header for nftw.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_FTW_NFTW_H
#define LLVM_LIBC_SRC_FTW_NFTW_H

#include "hdr/types/struct_FTW.h"
#include "hdr/types/struct_stat.h"
#include "src/__support/macros/config.h"

namespace LIBC_NAMESPACE_DECL {

int nftw(const char *dir_path,
         int (*fn)(const char *, const struct stat *, int, struct FTW *),
         int fd_limit, int flags);

} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC_FTW_NFTW_H
