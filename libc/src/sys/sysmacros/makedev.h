//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Implementation header for makedev.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_SYS_SYSMACROS_MAKEDEV_H
#define LLVM_LIBC_SRC_SYS_SYSMACROS_MAKEDEV_H

#include "hdr/types/dev_t.h"
#include "src/__support/macros/config.h"

namespace LIBC_NAMESPACE_DECL {

dev_t makedev(unsigned int major, unsigned int minor);

} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC_SYS_SYSMACROS_MAKEDEV_H
