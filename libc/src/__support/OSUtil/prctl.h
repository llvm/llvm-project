//===--------- Implementation header of the prctl function -----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC___SUPPORT_OSUTIL_PRCTL_H
#define LLVM_LIBC_SRC___SUPPORT_OSUTIL_PRCTL_H

#include "src/__support/error_or.h"
#include "src/__support/libc_errno.h"
#include "src/__support/macros/config.h"
#include <sys/prctl.h>

namespace LIBC_NAMESPACE_DECL {
namespace internal {

ErrorOr<int> prctl(int option, unsigned long arg2, unsigned long arg3,
                   unsigned long arg4, unsigned long arg5);

} // namespace internal
} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC___SUPPORT_OSUTIL_PRCTL_H
