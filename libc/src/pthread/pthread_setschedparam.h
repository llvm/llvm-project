//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Implementation header for pthread_setschedparam.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_PTHREAD_PTHREAD_SETSCHEDPARAM_H
#define LLVM_LIBC_SRC_PTHREAD_PTHREAD_SETSCHEDPARAM_H

#include "hdr/types/pthread_t.h"
#include "hdr/types/struct_sched_param.h"
#include "src/__support/macros/config.h"

namespace LIBC_NAMESPACE_DECL {

int pthread_setschedparam(pthread_t thread, int policy,
                          const struct sched_param *param);

} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC_PTHREAD_PTHREAD_SETSCHEDPARAM_H
