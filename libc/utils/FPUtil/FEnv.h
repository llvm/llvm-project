//===-- Utilities for manipulating floating point environment ---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_UTILS_FPUTIL_FENV_H
#define LLVM_LIBC_UTILS_FPUTIL_FENV_H

#ifdef __x86_64__
#include "x86_64/FEnv.h"
#else
#include "DummyFEnv.h"
#endif

#endif // LLVM_LIBC_UTILS_FPUTIL_FP_BITS_H
