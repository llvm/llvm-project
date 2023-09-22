//===-- Square root of IEEE 754 floating point numbers ----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC___SUPPORT_FPUTIL_SQRT_H
#define LLVM_LIBC_SRC___SUPPORT_FPUTIL_SQRT_H

#include "src/__support/macros/properties/architectures.h"

#if defined(LIBC_TARGET_ARCH_IS_X86_64)
#include "x86_64/sqrt.h"
#elif defined(LIBC_TARGET_ARCH_IS_AARCH64)
#include "aarch64/sqrt.h"
#elif defined(LIBC_TARGET_ARCH_IS_RISCV64)
#include "riscv64/sqrt.h"
#else
#include "generic/sqrt.h"

#endif
#endif // LLVM_LIBC_SRC___SUPPORT_FPUTIL_SQRT_H
