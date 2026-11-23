//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Implementation header for pthread_attr_getschedparam.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_PTHREAD_PTHREAD_ATTR_GETSCHEDPARAM_H
#define LLVM_LIBC_SRC_PTHREAD_PTHREAD_ATTR_GETSCHEDPARAM_H

#include "hdr/types/pthread_attr_t.h"
#include "hdr/types/struct_sched_param.h"
#include "src/__support/macros/config.h"

namespace LIBC_NAMESPACE_DECL {

int pthread_attr_getschedparam(const pthread_attr_t *__restrict attr,
                               struct sched_param *__restrict param);

} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC_PTHREAD_PTHREAD_ATTR_GETSCHEDPARAM_H
