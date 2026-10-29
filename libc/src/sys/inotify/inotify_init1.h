//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Implementation header for inotify_init1 function.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_SYS_INOTIFY_INOTIFY_INIT1_H
#define LLVM_LIBC_SRC_SYS_INOTIFY_INOTIFY_INIT1_H

#include "src/__support/macros/config.h"

namespace LIBC_NAMESPACE_DECL {

int inotify_init1(int flags);

} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC_SYS_INOTIFY_INOTIFY_INIT1_H
