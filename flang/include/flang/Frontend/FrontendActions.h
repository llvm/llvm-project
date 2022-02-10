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

#include "mlir/IR/BuiltinOps.h"
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

class InitOnlyAction : public FrontendAction {
  void ExecuteAction() override;
};

//===----------------------------------------------------------------------===//
// Prescan Actions
//===----------------------------------------------------------------------===//
class PrescanAction : public FrontendAction {
  void ExecuteAction() override = 0;
  bool BeginSourceFileAction() override;
};

class PrintPreprocessedAction : public PrescanAction {
  void ExecuteAction() override;
};

class DebugDumpProvenanceAction : public PrescanAction {
  void ExecuteAction() override;
};

class DebugDumpParsingLogAction : public PrescanAction {
  void ExecuteAction() override;
};

class DebugMeasureParseTreeAction : public PrescanAction {
  void ExecuteAction() override;
};

//===----------------------------------------------------------------------===//
// PrescanAndParse Actions
//===----------------------------------------------------------------------===//
class PrescanAndParseAction : public FrontendAction {
  void ExecuteAction() override = 0;
  bool BeginSourceFileAction() override;
};

class DebugUnparseNoSemaAction : public PrescanAndParseAction {
  void ExecuteAction() override;
};

class DebugDumpParseTreeNoSemaAction : public PrescanAndParseAction {
  void ExecuteAction() override;
};

//===----------------------------------------------------------------------===//
// PrescanAndSema Actions
//
// These actions will parse the input, run the semantic checks and execute
// their actions provided that no parsing or semantic errors were found.
//===----------------------------------------------------------------------===//
class PrescanAndSemaAction : public FrontendAction {

  void ExecuteAction() override = 0;
  bool BeginSourceFileAction() override;
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

class GetDefinitionAction : public PrescanAndSemaAction {
  void ExecuteAction() override;
};

class GetSymbolsSourcesAction : public PrescanAndSemaAction {
  void ExecuteAction() override;
};

class ParseSyntaxOnlyAction : public PrescanAndSemaAction {
  void ExecuteAction() override;
};

class PluginParseTreeAction : public PrescanAndSemaAction {
  void ExecuteAction() override = 0;
};

//===----------------------------------------------------------------------===//
// PrescanAndSemaDebug Actions
//
// These actions will parse the input, run the semantic checks and execute
// their actions regardless of whether any semantic errors are found.
//===----------------------------------------------------------------------===//
class PrescanAndSemaDebugAction : public FrontendAction {

  void ExecuteAction() override = 0;
  bool BeginSourceFileAction() override;
};

class DebugDumpAllAction : public PrescanAndSemaDebugAction {
  void ExecuteAction() override;
};

//===----------------------------------------------------------------------===//
// CodeGen Actions
//===----------------------------------------------------------------------===//
/// Abstract base class for actions that generate code (MLIR, LLVM IR, assembly
/// and machine code). Every action that inherits from this class will at
/// least run the prescanning, parsing, semantic checks and lower the parse
/// tree to an MLIR module.
class CodeGenAction : public FrontendAction {

  void ExecuteAction() override = 0;
  /// Runs prescan, parsing, sema and lowers to MLIR.
  bool BeginSourceFileAction() override;

protected:
  /// @name MLIR
  /// {
  std::unique_ptr<mlir::ModuleOp> mlirModule;
  std::unique_ptr<mlir::MLIRContext> mlirCtx;
  /// }
};

class EmitMLIRAction : public CodeGenAction {
  void ExecuteAction() override;
};

class EmitObjAction : public CodeGenAction {
  void ExecuteAction() override;
};

} // namespace Fortran::frontend

#endif // LLVM_FLANG_FRONTEND_FRONTENDACTIONS_H
