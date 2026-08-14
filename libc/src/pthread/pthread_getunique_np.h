//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Implementation header for pthread_getunique_np function.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_PTHREAD_PTHREAD_GETUNIQUE_NP_H
#define LLVM_LIBC_SRC_PTHREAD_PTHREAD_GETUNIQUE_NP_H

#include "include/llvm-libc-types/pthread_id_np_t.h"
#include "include/llvm-libc-types/pthread_t.h"
#include "src/__support/macros/config.h"

namespace LIBC_NAMESPACE_DECL {

int pthread_getunique_np(const pthread_t *__restrict thread,
                         pthread_id_np_t *__restrict id);

} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC_PTHREAD_PTHREAD_GETUNIQUE_NP_H
