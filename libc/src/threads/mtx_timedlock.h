//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Implementation header for mtx_timedlock function.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_THREADS_MTX_TIMEDLOCK_H
#define LLVM_LIBC_SRC_THREADS_MTX_TIMEDLOCK_H

#include "hdr/types/struct_timespec.h"
#include "src/__support/macros/config.h"
#include <threads.h>

namespace LIBC_NAMESPACE_DECL {

int mtx_timedlock(mtx_t *__restrict mutex,
                  const struct timespec *__restrict time_point);

} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC_THREADS_MTX_TIMEDLOCK_H
