//===-- include/flang/Support/Flags.h ---------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_SUPPORT_FLAGS_H_
#define FORTRAN_SUPPORT_FLAGS_H_

#include "llvm/Support/CommandLine.h"

extern llvm::cl::opt<bool> enableDelayedPrivatization;
extern llvm::cl::opt<bool> enableDelayedPrivatizationStaging;

#endif // FORTRAN_SUPPORT_FLAGS_H_
