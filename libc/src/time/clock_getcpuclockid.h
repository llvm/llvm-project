//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Implementation header for clock_getcpuclockid function.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_TIME_CLOCK_GETCPUCLOCKID_H
#define LLVM_LIBC_SRC_TIME_CLOCK_GETCPUCLOCKID_H

#include "hdr/types/clockid_t.h"
#include "hdr/types/pid_t.h"
#include "src/__support/macros/config.h"

namespace LIBC_NAMESPACE_DECL {

int clock_getcpuclockid(pid_t pid, clockid_t *clock_id);

} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC_TIME_CLOCK_GETCPUCLOCKID_H
