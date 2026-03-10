//===- TUSummaryExtractorFrontendAction.h -----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_ANALYSIS_SCALABLE_FRONTEND_TUSUMMARYEXTRACTORFRONTENDACTION_H
#define LLVM_CLANG_ANALYSIS_SCALABLE_FRONTEND_TUSUMMARYEXTRACTORFRONTENDACTION_H

#include "clang/Frontend/FrontendAction.h"
#include <memory>

namespace clang::ssaf {

/// Wraps the existing \c FrontendAction and injects the extractor
/// \c ASTConsumers into the pipeline after the ASTConsumers of the wrapped
/// action.
class TUSummaryExtractorFrontendAction final : public WrapperFrontendAction {
public:
  explicit TUSummaryExtractorFrontendAction(
      std::unique_ptr<FrontendAction> WrappedAction);
  ~TUSummaryExtractorFrontendAction();

protected:
  std::unique_ptr<ASTConsumer> CreateASTConsumer(CompilerInstance &CI,
                                                 StringRef InFile) override;
};

} // namespace clang::ssaf

#endif // LLVM_CLANG_ANALYSIS_SCALABLE_FRONTEND_TUSUMMARYEXTRACTORFRONTENDACTION_H
