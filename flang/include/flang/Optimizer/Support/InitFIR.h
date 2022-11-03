//===-- Optimizer/Support/InitFIR.h -----------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Coding style: https://mlir.llvm.org/getting_started/DeveloperGuide/
//
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_OPTIMIZER_SUPPORT_INITFIR_H
#define FORTRAN_OPTIMIZER_SUPPORT_INITFIR_H

#include "flang/Optimizer/Dialect/FIRDialect.h"
#include "flang/Optimizer/HLFIR/HLFIRDialect.h"
#include "mlir/Conversion/Passes.h"
#include "mlir/Dialect/Affine/Passes.h"
#include "mlir/Dialect/Complex/IR/Complex.h"
#include "mlir/InitAllDialects.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/LocationSnapshot.h"
#include "mlir/Transforms/Passes.h"

namespace fir::support {

#define FLANG_NONCODEGEN_DIALECT_LIST                                          \
  mlir::AffineDialect, FIROpsDialect, hlfir::hlfirDialect,                     \
      mlir::acc::OpenACCDialect, mlir::omp::OpenMPDialect,                     \
      mlir::scf::SCFDialect, mlir::arith::ArithDialect,                        \
      mlir::cf::ControlFlowDialect, mlir::func::FuncDialect,                   \
      mlir::vector::VectorDialect, mlir::math::MathDialect,                    \
      mlir::complex::ComplexDialect, mlir::DLTIDialect

// The definitive list of dialects used by flang.
#define FLANG_DIALECT_LIST                                                     \
  FLANG_NONCODEGEN_DIALECT_LIST, FIRCodeGenDialect, mlir::LLVM::LLVMDialect

inline void registerNonCodegenDialects(mlir::DialectRegistry &registry) {
  registry.insert<FLANG_NONCODEGEN_DIALECT_LIST>();
}

/// Register all the dialects used by flang.
inline void registerDialects(mlir::DialectRegistry &registry) {
  registry.insert<FLANG_DIALECT_LIST>();
}

inline void loadNonCodegenDialects(mlir::MLIRContext &context) {
  context.loadDialect<FLANG_NONCODEGEN_DIALECT_LIST>();
}

/// Forced load of all the dialects used by flang.  Lowering is not an MLIR
/// pass, but a producer of FIR and MLIR. It is therefore a requirement that the
/// dialects be preloaded to be able to build the IR.
inline void loadDialects(mlir::MLIRContext &context) {
  context.loadDialect<FLANG_DIALECT_LIST>();
}

/// Register the standard passes we use. This comes from registerAllPasses(),
/// but is a smaller set since we aren't using many of the passes found there.
inline void registerMLIRPassesForFortranTools() {
  mlir::registerCanonicalizerPass();
  mlir::registerCSEPass();
  mlir::registerAffineLoopFusionPass();
  mlir::registerLoopInvariantCodeMotionPass();
  mlir::registerLoopCoalescingPass();
  mlir::registerStripDebugInfoPass();
  mlir::registerPrintOpStatsPass();
  mlir::registerInlinerPass();
  mlir::registerSCCPPass();
  mlir::registerAffineScalarReplacementPass();
  mlir::registerSymbolDCEPass();
  mlir::registerLocationSnapshotPass();
  mlir::registerAffinePipelineDataTransferPass();

  mlir::registerAffineVectorizePass();
  mlir::registerAffineLoopUnrollPass();
  mlir::registerAffineLoopUnrollAndJamPass();
  mlir::registerSimplifyAffineStructuresPass();
  mlir::registerAffineLoopInvariantCodeMotionPass();
  mlir::registerAffineLoopTilingPass();
  mlir::registerAffineDataCopyGenerationPass();

  mlir::registerConvertAffineToStandardPass();
}

/// Register the interfaces needed to lower to LLVM IR.
void registerLLVMTranslation(mlir::MLIRContext &context);

} // namespace fir::support

#endif // FORTRAN_OPTIMIZER_SUPPORT_INITFIR_H
