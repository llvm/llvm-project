//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Implementation header for pthread_getthreadid_np.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_PTHREAD_PTHREAD_GETTHREADID_NP_H
#define LLVM_LIBC_SRC_PTHREAD_PTHREAD_GETTHREADID_NP_H

#include "include/llvm-libc-types/pthread_id_np_t.h"
#include "src/__support/macros/config.h"

namespace LIBC_NAMESPACE_DECL {

pthread_id_np_t pthread_getthreadid_np();

} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC_PTHREAD_PTHREAD_GETTHREADID_NP_H
