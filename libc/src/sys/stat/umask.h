//===-- Implementation header for umask -------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_SYS_STAT_UMASK_H
#define LLVM_LIBC_SRC_SYS_STAT_UMASK_H

#include "hdr/types/mode_t.h"
#include "src/__support/macros/config.h"

namespace LIBC_NAMESPACE_DECL {

mode_t umask(mode_t cmask);

} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC_SYS_STAT_UMASK_H
