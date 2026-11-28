//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Platform-specific File specialization dispatch header.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC___SUPPORT_FILE_PLATFORM_FILE_H
#define LLVM_LIBC_SRC___SUPPORT_FILE_PLATFORM_FILE_H

#if defined(__linux__)
#include "src/__support/File/linux/file.h"
#else
#error "Platform File specialization is not available on this target."
#endif

#endif // LLVM_LIBC_SRC___SUPPORT_FILE_PLATFORM_FILE_H
