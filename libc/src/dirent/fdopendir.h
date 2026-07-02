//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Declaration of the POSIX fdopendir function.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_DIRENT_FDOPENDIR_H
#define LLVM_LIBC_SRC_DIRENT_FDOPENDIR_H

#include "hdr/types/DIR.h"
#include "src/__support/macros/config.h"

namespace LIBC_NAMESPACE_DECL {

DIR *fdopendir(int fd);

} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC_DIRENT_FDOPENDIR_H
