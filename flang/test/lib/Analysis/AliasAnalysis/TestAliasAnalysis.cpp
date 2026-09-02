//===- TestAliasAnalysis.cpp - Test FIR lias analysis     -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/test/lib/Analysis/TestAliasAnalysis.h"
#include "mlir/Analysis/AliasAnalysis.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "flang/Optimizer/Analysis/AliasAnalysis.h"
#include "flang/Optimizer/Dialect/FIROps.h"

using namespace mlir;

namespace {

//===----------------------------------------------------------------------===//
// Testing AliasResult
//===----------------------------------------------------------------------===//

struct TestFIRAliasAnalysisPass
    : public test::TestAliasAnalysisBase,
      PassWrapper<TestFIRAliasAnalysisPass, OperationPass<>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestFIRAliasAnalysisPass)

  StringRef getArgument() const final { return "test-fir-alias-analysis"; }
  StringRef getDescription() const final {
    return "Test alias analysis results.";
  }
  void runOnOperation() override {
    mlir::AliasAnalysis aliasAnalysis(getOperation());
    aliasAnalysis.addAnalysisImplementation(fir::AliasAnalysis());
    runAliasAnalysisOnOperation(getOperation(), aliasAnalysis);
  }
};

//===----------------------------------------------------------------------===//
// Testing ModRefResult
//===----------------------------------------------------------------------===//

struct TestFIRAliasAnalysisModRefPass
    : public test::TestAliasAnalysisModRefBase,
      PassWrapper<TestFIRAliasAnalysisModRefPass, OperationPass<>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestFIRAliasAnalysisModRefPass)

  StringRef getArgument() const final {
    return "test-fir-alias-analysis-modref";
  }
  StringRef getDescription() const final {
    return "Test alias analysis ModRef results.";
  }
  void runOnOperation() override {
    mlir::AliasAnalysis aliasAnalysis(getOperation());
    aliasAnalysis.addAnalysisImplementation(fir::AliasAnalysis());
    runAliasAnalysisOnOperation(getOperation(), aliasAnalysis);
  }
};
//===----------------------------------------------------------------------===//
// Testing ModRefResult with MemoryEffectOpInterface attached to fir.call
//===----------------------------------------------------------------------===//

/// External model that attaches MemoryEffectOpInterface to fir::CallOp.
/// Calls to functions whose name starts with "test_pure" have no memory
/// effects; all other calls read and write DefaultResource.
struct TestCallMemoryEffectsModel
    : public MemoryEffectOpInterface::ExternalModel<TestCallMemoryEffectsModel,
          fir::CallOp> {
  void getEffects(Operation *op,
      llvm::SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
          &effects) const {
    auto call = cast<fir::CallOp>(op);
    if (auto callee = call.getCallee())
      if (callee->getLeafReference().getValue().starts_with("test_pure"))
        return;
    effects.emplace_back(
        MemoryEffects::Read::get(), SideEffects::DefaultResource::get());
    effects.emplace_back(
        MemoryEffects::Write::get(), SideEffects::DefaultResource::get());
  }
};

struct TestFIRAliasAnalysisModRefCallEffectsPass
    : public test::TestAliasAnalysisModRefBase,
      PassWrapper<TestFIRAliasAnalysisModRefCallEffectsPass, OperationPass<>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(
      TestFIRAliasAnalysisModRefCallEffectsPass)

  StringRef getArgument() const final {
    return "test-fir-alias-analysis-modref-call-effects";
  }
  StringRef getDescription() const final {
    return "Test alias analysis ModRef results with MemoryEffectOpInterface "
           "attached to fir.call.";
  }
  void runOnOperation() override {
    MLIRContext *ctx = &getContext();
    fir::CallOp::attachInterface<TestCallMemoryEffectsModel>(*ctx);

    AliasAnalysis aliasAnalysis(getOperation());
    aliasAnalysis.addAnalysisImplementation(fir::AliasAnalysis());
    runAliasAnalysisOnOperation(getOperation(), aliasAnalysis);
  }
};
} // namespace

//===----------------------------------------------------------------------===//
// Pass Registration
//===----------------------------------------------------------------------===//

namespace fir {
namespace test {
void registerTestFIRAliasAnalysisPass() {
  PassRegistration<TestFIRAliasAnalysisPass>();
  PassRegistration<TestFIRAliasAnalysisModRefPass>();
  PassRegistration<TestFIRAliasAnalysisModRefCallEffectsPass>();
}
} // namespace test
} // namespace fir