//===-- Convenient sanitizer annotations ----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_SUPPORT_SANITIZER_ANNOTATIONS_H
#define LLVM_LIBC_SRC_SUPPORT_SANITIZER_ANNOTATIONS_H

#if __has_feature(memory_sanitizer)
#include <sanitizer/msan_interface.h>
#define SANITIZER_MEMORY_INITIALIZED(addr, size) __msan_unpoison(addr, size)
#else
#define SANITIZER_MEMORY_INITIALIZED(ptr, size)
#endif

#endif // LLVM_LIBC_SRC_SUPPORT_SANITIZER_ANNOTATIONS_H
