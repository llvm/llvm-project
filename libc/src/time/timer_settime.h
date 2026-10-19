//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Implementation header for timer_settime function.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_TIME_TIMER_SETTIME_H
#define LLVM_LIBC_SRC_TIME_TIMER_SETTIME_H

#include "hdr/types/struct_itimerspec.h"
#include "hdr/types/timer_t.h"
#include "src/__support/macros/config.h"

namespace LIBC_NAMESPACE_DECL {

int timer_settime(timer_t timerid, int flags, const itimerspec *__restrict val,
                  itimerspec *__restrict old);

} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC_TIME_TIMER_SETTIME_H
