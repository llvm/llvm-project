//===- HLSL/FrontendActions.h -----------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_HLSL_FRONTEND_ACTIONS_H
#define LLVM_CLANG_HLSL_FRONTEND_ACTIONS_H

#include "clang/Frontend/FrontendAction.h"

namespace clang {

class HLSLFrontendAction : public WrapperFrontendAction {
protected:
  void ExecuteAction() override;

public:
  HLSLFrontendAction(std::unique_ptr<FrontendAction> WrappedAction);
};

} // namespace clang

#endif // LLVM_CLANG_HLSL_FRONTEND_ACTIONS_H
