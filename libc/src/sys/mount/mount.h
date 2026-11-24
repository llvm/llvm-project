//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Implementation header for mount.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_SYS_MOUNT_MOUNT_H
#define LLVM_LIBC_SRC_SYS_MOUNT_MOUNT_H

#include "src/__support/macros/config.h"

namespace LIBC_NAMESPACE_DECL {

int mount(const char *source, const char *target, const char *filesystemtype,
          unsigned long mountflags, const void *data);

} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC_SYS_MOUNT_MOUNT_H
