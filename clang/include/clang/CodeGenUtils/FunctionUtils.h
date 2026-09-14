//===--- FunctionUtils.h - Shared function emission queries -----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file holds the queries that both classic CodeGen and CIR CodeGen need
// while emitting a function body.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_CODEGENUTILS_FUNCTIONUTILS_H
#define LLVM_CLANG_CODEGENUTILS_FUNCTIONUTILS_H

#include "clang/Basic/CodeGenOptions.h"
#include "clang/Basic/LangOptions.h"

namespace clang::CodeGenUtils {

/// Decide whether we need to emit the lifetime markers.
bool shouldEmitLifetimeMarkers(const CodeGenOptions &CGOpts,
                               const LangOptions &LangOpts);

} // namespace clang::CodeGenUtils

#endif // LLVM_CLANG_CODEGENUTILS_FUNCTIONUTILS_H
