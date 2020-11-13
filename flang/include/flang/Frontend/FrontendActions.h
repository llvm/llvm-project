//===- FrontendActions.h -----------------------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_FLANG_FRONTEND_FRONTENDACTIONS_H
#define LLVM_FLANG_FRONTEND_FRONTENDACTIONS_H

#include "flang/Frontend/FrontendAction.h"

namespace Fortran::frontend {

//===----------------------------------------------------------------------===//
// Custom Consumer Actions
//===----------------------------------------------------------------------===//

class InputOutputTestAction : public FrontendAction {
  void ExecuteAction() override;
};

class PrintPreprocessedAction : public FrontendAction {
  void ExecuteAction() override;
};

} // namespace Fortran::frontend

#endif // LLVM_FLANG_FRONTEND_FRONTENDACTIONS_H
