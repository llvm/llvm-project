//===- BoundsSafetyArgs.h ---------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef LLVM_CLANG_BASIC_BOUNDS_SAFETY_ARGS_H
#define LLVM_CLANG_BASIC_BOUNDS_SAFETY_ARGS_H
#include "clang/Basic/Diagnostic.h"
#include "clang/Basic/LangOptions.h"
#include "llvm/Option/ArgList.h"

namespace clang {
namespace driver {

LangOptions::BoundsSafetyNewChecksMaskIntTy
ParseBoundsSafetyNewChecksMaskFromArgs(const llvm::opt::ArgList &Args,
                                       DiagnosticsEngine *Diags);
} // namespace driver
} // namespace clang

#endif
