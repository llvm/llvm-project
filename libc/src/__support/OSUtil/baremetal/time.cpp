//===---------- Baremetal implementation of time utils ----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "time.h"

#include "src/__support/macros/config.h"

namespace LIBC_NAMESPACE_DECL {

extern "C" int __llvm_libc_time_utc_get(struct timespec *ts);

int time_utc_get(struct timespec *ts) {
  return __llvm_libc_time_utc_get(ts);
}

} // namespace LIBC_NAMESPACE_DECL
