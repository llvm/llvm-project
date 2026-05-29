//===-- Internal header for abort -----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_STDLIB_ABORT_UTILS_H
#define LLVM_LIBC_SRC_STDLIB_ABORT_UTILS_H

#include "src/__support/macros/properties/architectures.h"

#if defined(LIBC_TARGET_ARCH_IS_GPU)
#include "src/stdlib/gpu/abort_utils.h"
#elif defined(__linux__)
#include "src/stdlib/linux/abort_utils.h"
#elif defined(__ELF__)
// TODO:ELF detection logic is borrowed from io.h (as we are still missing
// LIBC_TARGET_OS_IS_BAREMETAL).
#include "src/stdlib/baremetal/abort_utils.h"
#endif

#endif // LLVM_LIBC_SRC_STDLIB_ABORT_UTILS_H
