//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Implementation header of scandir
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_DIRENT_SCANDIR_H
#define LLVM_LIBC_SRC_DIRENT_SCANDIR_H

#include "hdr/types/struct_dirent.h"
#include "src/__support/macros/config.h"

namespace LIBC_NAMESPACE_DECL {

int scandir(const char *dir, struct dirent ***namelist,
            int (*sel)(const struct dirent *),
            int (*compar)(const struct dirent **, const struct dirent **));

} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC_DIRENT_SCANDIR_H
