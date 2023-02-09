//===-- Floating point environment manipulation functions -------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_SUPPORT_FPUTIL_FENVIMPL_H
#define LLVM_LIBC_SRC_SUPPORT_FPUTIL_FENVIMPL_H

#include "src/__support/macros/attributes.h" // LIBC_INLINE
#include "src/__support/macros/properties/architectures.h"

#if defined(LIBC_TARGET_ARCH_IS_AARCH64)
#if defined(__APPLE__)
#include "aarch64/fenv_darwin_impl.h"
#else
#include "aarch64/FEnvImpl.h"
#endif
#elif defined(LIBC_TARGET_ARCH_IS_X86)
#include "x86_64/FEnvImpl.h"
#else
#include <fenv.h>

namespace __llvm_libc {
namespace fputil {

// All dummy functions silently succeed.

LIBC_INLINE int clear_except(int) { return 0; }

LIBC_INLINE int test_except(int) { return 0; }

LIBC_INLINE int set_except(int) { return 0; }

LIBC_INLINE int raise_except(int) { return 0; }

LIBC_INLINE int get_round() { return FE_TONEAREST; }

LIBC_INLINE int set_round(int) { return 0; }

LIBC_INLINE int get_env(fenv_t *) { return 0; }

LIBC_INLINE int set_env(const fenv_t *) { return 0; }

} // namespace fputil
} // namespace __llvm_libc
#endif

#endif // LLVM_LIBC_SRC_SUPPORT_FPUTIL_FENVIMPL_H
