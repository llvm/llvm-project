//===-- ClangDoc.h - ClangDoc -----------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file exposes a method to craete the FrontendActionFactory for the
// clang-doc tool. The factory runs the clang-doc mapper on a given set of
// source code files, storing the results key-value pairs in its
// ExecutionContext.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_DOC_CLANGDOC_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_DOC_CLANGDOC_H

#include "clang/Tooling/Execution.h"
#include "clang/Tooling/StandaloneExecution.h"
#include "clang/Tooling/Tooling.h"

namespace clang {
namespace doc {

std::unique_ptr<tooling::FrontendActionFactory>
newMapperActionFactory(tooling::ExecutionContext *ECtx);

} // namespace doc
} // namespace clang

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_DOC_CLANGDOC_H
