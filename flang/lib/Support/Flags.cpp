//===-- lib/Support/Flags.cpp ---------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang/Support/Flags.h"

llvm::cl::opt<bool> enableDelayedPrivatization("enable-delayed-privatization",
    llvm::cl::desc(
        "Emit private/local variables as clauses/specifiers on MLIR ops."),
    llvm::cl::init(true));

llvm::cl::opt<bool> enableDelayedPrivatizationStaging(
    "enable-delayed-privatization-staging",
    llvm::cl::desc("For partially supported constructs, emit private/local "
                   "variables as clauses/specifiers on MLIR ops."),
    llvm::cl::init(false));
