//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Internal implementation header for ftw and nftw.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_FTW_FTW_IMPL_H
#define LLVM_LIBC_SRC_FTW_FTW_IMPL_H

#include "hdr/types/__ftw_func_t.h"
#include "hdr/types/__nftw_func_t.h"
#include "src/__support/CPP/expected.h"
#include "src/__support/macros/config.h"

namespace LIBC_NAMESPACE_DECL {
namespace internal {

cpp::expected<int, int> ftw_impl(const char *dir_path, __ftw_func_t fn,
                                 int fd_limit);

cpp::expected<int, int> nftw_impl(const char *dir_path, __nftw_func_t fn,
                                  int fd_limit, int flags);

} // namespace internal
} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC_FTW_FTW_IMPL_H
