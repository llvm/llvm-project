//===- SourceTransformationFrontendAction.h ---------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_SCALABLESTATICANALYSIS_FRONTEND_SOURCETRANSFORMATIONFRONTENDACTION_H
#define LLVM_CLANG_SCALABLESTATICANALYSIS_FRONTEND_SOURCETRANSFORMATIONFRONTENDACTION_H

#include "clang/Frontend/FrontendAction.h"
#include <memory>

namespace clang::ssaf {

/// Wraps the existing \c FrontendAction and runs the source-transformation
/// pipeline alongside it. The transformation consumes a \c WPASuite read from
/// \c SSAFOptions::GlobalScopeAnalysisResult and emits source edits and a
/// transformation report to the configured output files.
class SourceTransformationFrontendAction final : public WrapperFrontendAction {
public:
  explicit SourceTransformationFrontendAction(
      std::unique_ptr<FrontendAction> WrappedAction);
  ~SourceTransformationFrontendAction();

protected:
  std::unique_ptr<ASTConsumer> CreateASTConsumer(CompilerInstance &CI,
                                                 StringRef InFile) override;
};

} // namespace clang::ssaf

#endif // LLVM_CLANG_SCALABLESTATICANALYSIS_FRONTEND_SOURCETRANSFORMATIONFRONTENDACTION_H
