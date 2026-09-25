//===- TestOpenACCSupport.cpp ---------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/OpenACC/Analysis/OpenACCSupport.h"
#include "mlir/Dialect/OpenACC/OpenACC.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "flang/Optimizer/Dialect/FIRDialect.h"
#include "flang/Optimizer/HLFIR/HLFIRDialect.h"
#include "flang/Optimizer/OpenACC/Analysis/FIROpenACCSupportAnalysis.h"

using namespace mlir;

namespace {

struct TestFIROpenACCSupport
    : public PassWrapper<TestFIROpenACCSupport, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestFIROpenACCSupport)

  StringRef getArgument() const final { return "test-fir-openacc-support"; }
  StringRef getDescription() const final {
    return "Test FIR implementation of the OpenACCSupport analysis.";
  }
  void getDependentDialects(::mlir::DialectRegistry &registry) const override {
    registry.insert<fir::FIROpsDialect, hlfir::hlfirDialect,
        mlir::acc::OpenACCDialect>();
  }

  void runOnOperation() override {
    auto &support = getAnalysis<mlir::acc::OpenACCSupport>();
    support.setImplementation(fir::acc::FIROpenACCSupportAnalysis());

    mlir::acc::VariableNameConfig demangled;
    mlir::acc::VariableNameConfig mangled;
    mangled.preferDemangledName = false;

    getOperation().walk([&](Operation *op) {
      // Only the operations marked with test.var_name are reported, so that a
      // test states which value it asks the name of.
      if (!op->hasAttr("test.var_name"))
        return;
      for (Value result : op->getResults()) {
        llvm::errs() << "Visiting: " << *op << "\n";
        llvm::errs() << "\tDemangled name: \""
                     << support.getVariableName(result, demangled) << "\"\n";
        llvm::errs() << "\tMangled name: \""
                     << support.getVariableName(result, mangled) << "\"\n";
      }
    });
  }
};
} // namespace

//===----------------------------------------------------------------------===//
// Pass Registration
//===----------------------------------------------------------------------===//

namespace fir {
namespace test {
void registerTestFIROpenACCSupportPass() {
  PassRegistration<TestFIROpenACCSupport>();
}
} // namespace test
} // namespace fir
