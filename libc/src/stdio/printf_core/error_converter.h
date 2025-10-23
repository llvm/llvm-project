//===-- Error converter for printf ------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_STDIO_PRINTF_CORE_ERROR_CONVERTER_H
#define LLVM_LIBC_SRC_STDIO_PRINTF_CORE_ERROR_CONVERTER_H

#include "src/__support/macros/properties/architectures.h"

// Maps internal errors to the available errnos on the platform.
#if defined(__linux__)
#include "linux/error_converter.h"
#elif defined(__ELF__)
// TODO: Ideally we would have LIBC_TARGET_OS_IS_BAREMETAL.
#include "baremetal/error_converter.h"
#endif

#endif // LLVM_LIBC_SRC_STDIO_PRINTF_CORE_ERROR_CONVERTER_H
