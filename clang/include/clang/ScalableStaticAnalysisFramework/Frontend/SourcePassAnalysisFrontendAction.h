//===- SourcePassAnalysisFrontendAction.h -----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_SCALABLESTATICANALYSISFRAMEWORK_FRONTEND_SOURCEPASSANALYSISFRONTENDACTION_H
#define LLVM_CLANG_SCALABLESTATICANALYSISFRAMEWORK_FRONTEND_SOURCEPASSANALYSISFRONTENDACTION_H

#include "clang/Frontend/FrontendAction.h"
#include <memory>

namespace clang::ssaf {

/// Wraps the existing \c FrontendAction and injects the source-pass analysis
/// \c ASTConsumers into the pipeline after the ASTConsumers of the wrapped
/// action.
class SourcePassAnalysisFrontendAction final : public WrapperFrontendAction {
public:
  explicit SourcePassAnalysisFrontendAction(
      std::unique_ptr<FrontendAction> WrappedAction);
  ~SourcePassAnalysisFrontendAction();

protected:
  std::unique_ptr<ASTConsumer> CreateASTConsumer(CompilerInstance &CI,
                                                 StringRef InFile) override;
};

} // namespace clang::ssaf

#endif // LLVM_CLANG_SCALABLESTATICANALYSISFRAMEWORK_FRONTEND_SOURCEPASSANALYSISFRONTENDACTION_H
