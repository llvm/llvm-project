//===-- ClangDoc.h - ClangDoc -----------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file exposes a method to create the FrontendActionFactory for the
// clang-doc tool. The factory runs the clang-doc mapper on a given set of
// source code files, storing the results key-value pairs in its
// ExecutionContext.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_DOC_CLANGDOC_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_DOC_CLANGDOC_H

#include "Representation.h"
#include "clang/Tooling/Execution.h"
#include "clang/Tooling/StandaloneExecution.h"
#include "clang/Tooling/Tooling.h"

namespace clang {
namespace doc {

std::unique_ptr<tooling::FrontendActionFactory>
newMapperActionFactory(ClangDocContext CDCtx);

} // namespace doc
} // namespace clang

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_DOC_CLANGDOC_H
