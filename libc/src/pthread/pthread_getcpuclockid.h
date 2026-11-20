//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Implementation header for pthread_getcpuclockid.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_PTHREAD_PTHREAD_GETCPUCLOCKID_H
#define LLVM_LIBC_SRC_PTHREAD_PTHREAD_GETCPUCLOCKID_H

#include "hdr/types/clockid_t.h"
#include "hdr/types/pthread_t.h"
#include "src/__support/macros/config.h"

namespace LIBC_NAMESPACE_DECL {

int pthread_getcpuclockid(pthread_t thread, clockid_t *clock_id);

} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC_PTHREAD_PTHREAD_GETCPUCLOCKID_H
