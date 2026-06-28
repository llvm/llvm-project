//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Header for basename.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_LIBGEN_BASENAME_H
#define LLVM_LIBC_SRC_LIBGEN_BASENAME_H

#include "src/__support/macros/config.h"

namespace LIBC_NAMESPACE_DECL {

/// Return the last component of a pathname.
///
/// \param path Pointer to the null-terminated pathname string.
/// \return Pointer to the last component of path, or "." if path is null or
/// empty, or "/" if path is all slashes.
char *basename(char *path);

} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC_LIBGEN_BASENAME_H
