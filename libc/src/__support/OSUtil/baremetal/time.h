//===---------- Baremetal implementation of time utils ----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC___SUPPORT_OSUTIL_BAREMETAL_TIME_H
#define LLVM_LIBC_SRC___SUPPORT_OSUTIL_BAREMETAL_TIME_H

#include "include/llvm-libc-types/struct_timespec.h"
#include "src/__support/macros/config.h"

namespace LIBC_NAMESPACE_DECL {
namespace internal {

bool timespec_get_utc(struct timespec *ts);

} // namespace internal
} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC___SUPPORT_OSUTIL_BAREMETAL_IO_H
