//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Implementation header for __pthread_cleanup_push.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_PTHREAD___PTHREAD_CLEANUP_PUSH_H
#define LLVM_LIBC_SRC_PTHREAD___PTHREAD_CLEANUP_PUSH_H

#include "hdr/types/struct___pthread_cleanup_frame.h"
#include "src/__support/macros/config.h"

namespace LIBC_NAMESPACE_DECL {

void __pthread_cleanup_push(struct __pthread_cleanup_frame *frame,
                            void (*routine)(void *), void *arg);

} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC_PTHREAD___PTHREAD_CLEANUP_PUSH_H
