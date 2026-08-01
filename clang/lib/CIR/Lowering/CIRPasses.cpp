//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements machinery for any CIR <-> CIR passes used by clang.
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/PassManager.h"
#include "clang/AST/ASTContext.h"
#include "clang/Basic/TargetInfo.h"
#include "clang/CIR/Dialect/Passes.h"
#include "llvm/Support/TimeProfiler.h"
#include "llvm/TargetParser/Triple.h"

namespace cir {

/// Map a target triple to the ABI target that drives CallConvLowering.
/// Returns None for targets whose calling convention is not yet implemented.
static CallConvTarget getCallConvTarget(const llvm::Triple &triple) {
  if (triple.getArch() == llvm::Triple::x86_64)
    return CallConvTarget::X86_64;
  return CallConvTarget::None;
}

mlir::LogicalResult
runCIRToCIRPasses(mlir::ModuleOp theModule, mlir::MLIRContext &mlirContext,
                  clang::ASTContext &astContext, bool enableVerifier,
                  bool enableIdiomRecognizer, bool enableCIRSimplify,
                  bool enableLibOpt, llvm::StringRef libOptOptions,
                  bool enableCallConvLowering) {

  llvm::TimeTraceScope scope("CIR To CIR Passes");

  mlir::PassManager pm(&mlirContext);
  pm.addPass(mlir::createCIRCanonicalizePass());

  if (enableCIRSimplify)
    pm.addPass(mlir::createCIRSimplifyPass());

  if (enableIdiomRecognizer)
    pm.addPass(mlir::createIdiomRecognizerPass());

  if (enableLibOpt) {
    auto libOptPass = mlir::createLibOptPass();
    auto errorHandler = [](const llvm::Twine &) -> mlir::LogicalResult {
      return mlir::LogicalResult::failure();
    };

    if (libOptPass->initializeOptions(libOptOptions, errorHandler).failed())
      return mlir::failure();

    pm.addPass(std::move(libOptPass));
  }

  pm.addPass(mlir::createTargetLoweringPass());
  pm.addPass(mlir::createCXXABILoweringPass());

  if (enableCallConvLowering) {
    // CallConvLowering rewrites signatures and call sites using the classifier,
    // so it must run after CXXABILowering has lowered C++ ABI types to plain
    // records the classifier can handle.  Only the x86_64 System V classifier
    // is implemented; other targets are left unchanged.
    CallConvTarget target =
        getCallConvTarget(astContext.getTargetInfo().getTriple());
    if (target != CallConvTarget::None)
      pm.addPass(mlir::createCallConvLoweringPass(
          target, llvm::abi::X86AVXABILevel::None));
  }

  pm.addPass(mlir::createLoweringPreparePass(&astContext));

  pm.enableVerifier(enableVerifier);
  (void)mlir::applyPassManagerCLOptions(pm);
  return pm.run(theModule);
}

} // namespace cir

namespace mlir {

void populateCIRPreLoweringPasses(OpPassManager &pm) {
  pm.addPass(createHoistAllocasPass());
  pm.addPass(createCIRFlattenCFGPass());
  pm.addPass(createCIREHABILoweringPass());
  pm.addPass(createGotoSolverPass());
}

} // namespace mlir
