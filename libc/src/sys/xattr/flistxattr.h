//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Function declaration of flistxattr.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_SYS_XATTR_LINUX_FLISTXATTR_H
#define LLVM_LIBC_SRC_SYS_XATTR_LINUX_FLISTXATTR_H

#include "hdr/types/size_t.h"
#include "hdr/types/ssize_t.h"
#include "src/__support/macros/config.h"

namespace LIBC_NAMESPACE_DECL {

ssize_t flistxattr(int fd, char *list, size_t size);

} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC_SYS_XATTR_LINUX_FLISTXATTR_H
