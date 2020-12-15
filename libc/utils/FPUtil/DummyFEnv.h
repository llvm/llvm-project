//===-- Dummy floating point environment manipulation functins --*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_UTILS_FPUTIL_DUMMY_FENV_H
#define LLVM_LIBC_UTILS_FPUTIL_DUMMY_FENV_H

#include <fenv.h>
#include <math.h>

namespace __llvm_libc {
namespace fputil {

// All dummy functions silently succeed.

static inline int clearExcept(int) { return 0; }

static inline int testExcept(int) { return 0; }

static inline int raiseExcept(int) { return 0; }

static inline int getRound() { return FE_TONEAREST; }

static inline int setRound(int) { return 0; }

} // namespace fputil
} // namespace __llvm_libc

#endif // LLVM_LIBC_UTILS_FPUTIL_DUMMY_FENV_H
