//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Implementation header for clock_nanosleep.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_TIME_CLOCK_NANOSLEEP_H
#define LLVM_LIBC_SRC_TIME_CLOCK_NANOSLEEP_H

#include "hdr/types/clockid_t.h"
#include "hdr/types/struct_timespec.h"
#include "src/__support/macros/config.h"

namespace LIBC_NAMESPACE_DECL {

int clock_nanosleep(clockid_t clockid, int flags, const timespec *req,
                    timespec *rem);

} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC_TIME_CLOCK_NANOSLEEP_H
