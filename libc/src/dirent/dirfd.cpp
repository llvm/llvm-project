//===-- Implementation of dirfd -------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "dirfd.h"

#include "src/__support/File/dir.h"
#include "src/__support/common.h"
#include "src/__support/macros/config.h"

#include <dirent.h>

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(int, dirfd, (::DIR * dir)) {
  auto *d = reinterpret_cast<LIBC_NAMESPACE::Dir *>(dir);
  return d->getfd();
}

} // namespace LIBC_NAMESPACE_DECL
