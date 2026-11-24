//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Implementation header for thrd_sleep.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_THREADS_THRD_SLEEP_H
#define LLVM_LIBC_SRC_THREADS_THRD_SLEEP_H

#include "hdr/types/struct_timespec.h"
#include "src/__support/macros/config.h"

namespace LIBC_NAMESPACE_DECL {

int thrd_sleep(const struct timespec *duration, struct timespec *remaining);

} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC_THREADS_THRD_SLEEP_H
