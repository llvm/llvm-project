//===-- Floating point environment manipulation functions -------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_SUPPORT_FPUTIL_DUMMY_FENVIMPL_H
#define LLVM_LIBC_SRC_SUPPORT_FPUTIL_DUMMY_FENVIMPL_H

#include "src/__support/architectures.h"

#if defined(LLVM_LIBC_ARCH_AARCH64)
#include "aarch64/FEnvImpl.h"
#elif defined(LLVM_LIBC_ARCH_X86)
#include "x86_64/FEnvImpl.h"
#else
#include <fenv.h>
#include <math.h>

namespace __llvm_libc {
namespace fputil {

// All dummy functions silently succeed.

static inline int clear_except(int) { return 0; }

static inline int test_except(int) { return 0; }

static inline int set_except(int) { return 0; }

static inline int raise_except(int) { return 0; }

static inline int get_round() { return FE_TONEAREST; }

static inline int set_round(int) { return 0; }

static inline int get_env(fenv_t *) { return 0; }

static inline int set_env(const fenv_t *) { return 0; }

} // namespace fputil
} // namespace __llvm_libc
#endif

#endif // LLVM_LIBC_SRC_SUPPORT_FPUTIL_DUMMY_FENVIMPL_H
