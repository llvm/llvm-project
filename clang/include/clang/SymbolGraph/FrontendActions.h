//===- SymbolGraph/FrontendActions.h -----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// \brief Defines SymbolGraph frontend actions.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_SYMBOLGRAPH_FRONTEND_ACTIONS_H
#define LLVM_CLANG_SYMBOLGRAPH_FRONTEND_ACTIONS_H

#include "clang/Frontend/FrontendAction.h"

namespace clang {

class ExtractAPIAction : public ASTFrontendAction {
protected:
  std::unique_ptr<ASTConsumer> CreateASTConsumer(CompilerInstance &CI,
                                                 StringRef InFile) override;

public:
  static std::unique_ptr<llvm::raw_pwrite_stream>
  CreateOutputFile(CompilerInstance &CI, StringRef InFile);
};

} // namespace clang

#endif // LLVM_CLANG_SYMBOLGRAPH_FRONTEND_ACTIONS_H
