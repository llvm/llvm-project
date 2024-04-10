//===- unittests/Frontend/CodeGenActionTest.cpp --- FrontendAction tests --===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Unit tests for CodeGenAction.
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/Builders.h"
#include "flang/Frontend/CompilerInstance.h"
#include "flang/Frontend/FrontendActions.h"
#include "flang/Frontend/TextDiagnosticPrinter.h"

#include "gtest/gtest.h"

#include <memory>

using namespace Fortran::frontend;

namespace test {
class DummyDialect : public ::mlir::Dialect {
  explicit DummyDialect(::mlir::MLIRContext *context)
      : ::mlir::Dialect(getDialectNamespace(), context,
            ::mlir::TypeID::get<DummyDialect>()) {
    initialize();
  }

  void initialize();
  friend class ::mlir::MLIRContext;

public:
  ~DummyDialect() override = default;
  static constexpr ::llvm::StringLiteral getDialectNamespace() {
    return ::llvm::StringLiteral("dummy");
  }
};

namespace dummy {
class FakeOp : public ::mlir::Op<FakeOp> {
public:
  using Op::Op;

  static llvm::StringRef getOperationName() { return "dummy.fake"; }

  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() { return {}; }

  static void build(
      ::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState) {}
};
} // namespace dummy
} // namespace test

MLIR_DECLARE_EXPLICIT_TYPE_ID(::test::DummyDialect)
MLIR_DEFINE_EXPLICIT_TYPE_ID(::test::DummyDialect)

namespace test {

void DummyDialect::initialize() { addOperations<::test::dummy::FakeOp>(); }
} // namespace test

// A test CodeGenAction to verify that we gracefully handle failure to convert
// from MLIR to LLVM IR.
class LLVMConversionFailureCodeGenAction : public CodeGenAction {
public:
  LLVMConversionFailureCodeGenAction()
      : CodeGenAction(BackendActionTy::Backend_EmitLL) {
    mlirCtx = std::make_unique<mlir::MLIRContext>();
    mlirCtx->loadDialect<test::DummyDialect>();

    mlir::Location loc(mlir::UnknownLoc::get(mlirCtx.get()));
    mlirModule =
        std::make_unique<mlir::ModuleOp>(mlir::ModuleOp::create(loc, "mod"));

    mlir::OpBuilder builder(mlirCtx.get());
    builder.setInsertionPointToStart(&mlirModule->getRegion().front());
    // Create a fake op to trip conversion to LLVM.
    builder.create<test::dummy::FakeOp>(loc);

    llvmCtx = std::make_unique<llvm::LLVMContext>();
  }
};

TEST(CodeGenAction, GracefullyHandleLLVMConversionFailure) {
  std::string diagnosticOutput;
  llvm::raw_string_ostream diagnosticsOS(diagnosticOutput);
  auto diagPrinter = std::make_unique<Fortran::frontend::TextDiagnosticPrinter>(
      diagnosticsOS, new clang::DiagnosticOptions());

  CompilerInstance ci;
  ci.createDiagnostics(diagPrinter.get(), /*ShouldOwnClient=*/false);
  ci.setInvocation(std::make_shared<CompilerInvocation>());
  ci.setOutputStream(std::make_unique<llvm::raw_null_ostream>());
  ci.getInvocation().getCodeGenOpts().OptimizationLevel = 0;

  FrontendInputFile file("/dev/null", InputKind());

  LLVMConversionFailureCodeGenAction action;
  action.setInstance(&ci);
  action.setCurrentInput(file);

  consumeError(action.execute());
  ASSERT_EQ(diagnosticsOS.str(),
      "error: Lowering to LLVM IR failed\n"
      "error: failed to create the LLVM module\n");
}
