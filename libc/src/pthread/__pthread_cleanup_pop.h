//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Implementation header for __pthread_cleanup_pop.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_PTHREAD___PTHREAD_CLEANUP_POP_H
#define LLVM_LIBC_SRC_PTHREAD___PTHREAD_CLEANUP_POP_H

#include "hdr/types/struct___pthread_cleanup_frame.h"
#include "src/__support/macros/config.h"

namespace LIBC_NAMESPACE_DECL {

void __pthread_cleanup_pop(struct __pthread_cleanup_frame *frame, int execute);

} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC_PTHREAD___PTHREAD_CLEANUP_POP_H
