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
#include "flang/Semantics/semantics.h"
#include <memory>

namespace Fortran::frontend {

// TODO: This is a copy from f18.cpp. It doesn't really belong here and should
// be moved to a more suitable place in future.
struct MeasurementVisitor {
  template <typename A> bool Pre(const A &) { return true; }
  template <typename A> void Post(const A &) {
    ++objects;
    bytes += sizeof(A);
  }
  size_t objects{0}, bytes{0};
};

//===----------------------------------------------------------------------===//
// Custom Consumer Actions
//===----------------------------------------------------------------------===//

class InputOutputTestAction : public FrontendAction {
  void ExecuteAction() override;
};

class EmitObjAction : public FrontendAction {
  void ExecuteAction() override;
};

//===----------------------------------------------------------------------===//
// Prescan Actions
//===----------------------------------------------------------------------===//
class PrescanAction : public FrontendAction {
  void ExecuteAction() override = 0;
  bool BeginSourceFileAction(CompilerInstance &ci) override;
};

class PrintPreprocessedAction : public PrescanAction {
  void ExecuteAction() override;
};

class DebugDumpProvenanceAction : public PrescanAction {
  void ExecuteAction() override;
};

class DebugMeasureParseTreeAction : public PrescanAction {
  void ExecuteAction() override;
};

//===----------------------------------------------------------------------===//
// PrescanAndSema Actions
//===----------------------------------------------------------------------===//
class PrescanAndSemaAction : public FrontendAction {
  std::unique_ptr<Fortran::semantics::Semantics> semantics_;

  void ExecuteAction() override = 0;
  bool BeginSourceFileAction(CompilerInstance &ci) override;

public:
  Fortran::semantics::Semantics &semantics() { return *semantics_; }
  const Fortran::semantics::Semantics &semantics() const { return *semantics_; }

  void setSemantics(std::unique_ptr<Fortran::semantics::Semantics> semantics) {
    semantics_ = std::move(semantics);
  }
};

class DebugUnparseWithSymbolsAction : public PrescanAndSemaAction {
  void ExecuteAction() override;
};

class DebugUnparseAction : public PrescanAndSemaAction {
  void ExecuteAction() override;
};

class DebugDumpSymbolsAction : public PrescanAndSemaAction {
  void ExecuteAction() override;
};

class DebugDumpParseTreeAction : public PrescanAndSemaAction {
  void ExecuteAction() override;
};

class DebugPreFIRTreeAction : public PrescanAndSemaAction {
  void ExecuteAction() override;
};

class ParseSyntaxOnlyAction : public PrescanAndSemaAction {
  void ExecuteAction() override;
};

} // namespace Fortran::frontend

#endif // LLVM_FLANG_FRONTEND_FRONTENDACTIONS_H
