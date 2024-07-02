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

#include "flang/Optimizer/Dialect/CUF/CUFDialect.h"
#include "flang/Optimizer/Dialect/FIRDialect.h"
#include "flang/Optimizer/HLFIR/HLFIRDialect.h"
#include "mlir/Conversion/Passes.h"
#include "mlir/Dialect/Affine/Passes.h"
#include "mlir/Dialect/Complex/IR/Complex.h"
#include "mlir/Dialect/Func/Extensions/InlinerExtension.h"
#include "mlir/Dialect/OpenACC/Transforms/Passes.h"
#include "mlir/InitAllDialects.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/LocationSnapshot.h"
#include "mlir/Transforms/Passes.h"

namespace fir::support {

#define FLANG_NONCODEGEN_DIALECT_LIST                                          \
  mlir::affine::AffineDialect, FIROpsDialect, hlfir::hlfirDialect,             \
      mlir::acc::OpenACCDialect, mlir::omp::OpenMPDialect,                     \
      mlir::scf::SCFDialect, mlir::arith::ArithDialect,                        \
      mlir::cf::ControlFlowDialect, mlir::func::FuncDialect,                   \
      mlir::vector::VectorDialect, mlir::math::MathDialect,                    \
      mlir::complex::ComplexDialect, mlir::DLTIDialect, cuf::CUFDialect

#define FLANG_CODEGEN_DIALECT_LIST FIRCodeGenDialect, mlir::LLVM::LLVMDialect

// The definitive list of dialects used by flang.
#define FLANG_DIALECT_LIST                                                     \
  FLANG_NONCODEGEN_DIALECT_LIST, FLANG_CODEGEN_DIALECT_LIST

inline void registerNonCodegenDialects(mlir::DialectRegistry &registry) {
  registry.insert<FLANG_NONCODEGEN_DIALECT_LIST>();
  mlir::func::registerInlinerExtension(registry);
}

/// Register all the dialects used by flang.
inline void registerDialects(mlir::DialectRegistry &registry) {
  registerNonCodegenDialects(registry);
  registry.insert<FLANG_CODEGEN_DIALECT_LIST>();
}

// Register FIR Extensions
inline void addFIRExtensions(mlir::DialectRegistry &registry,
                             bool addFIRInlinerInterface = true) {
  if (addFIRInlinerInterface)
    addFIRInlinerExtension(registry);
  addFIRToLLVMIRExtension(registry);
}

inline void loadNonCodegenDialects(mlir::MLIRContext &context) {
  mlir::DialectRegistry registry;
  registerNonCodegenDialects(registry);
  context.appendDialectRegistry(registry);

  context.loadDialect<FLANG_NONCODEGEN_DIALECT_LIST>();
}

/// Forced load of all the dialects used by flang.  Lowering is not an MLIR
/// pass, but a producer of FIR and MLIR. It is therefore a requirement that the
/// dialects be preloaded to be able to build the IR.
inline void loadDialects(mlir::MLIRContext &context) {
  mlir::DialectRegistry registry;
  registerDialects(registry);
  context.appendDialectRegistry(registry);

  context.loadDialect<FLANG_DIALECT_LIST>();
}

/// Register the standard passes we use. This comes from registerAllPasses(),
/// but is a smaller set since we aren't using many of the passes found there.
inline void registerMLIRPassesForFortranTools() {
  mlir::acc::registerOpenACCPasses();
  mlir::registerCanonicalizerPass();
  mlir::registerCSEPass();
  mlir::affine::registerAffineLoopFusionPass();
  mlir::registerLoopInvariantCodeMotionPass();
  mlir::affine::registerLoopCoalescingPass();
  mlir::registerStripDebugInfoPass();
  mlir::registerPrintOpStatsPass();
  mlir::registerInlinerPass();
  mlir::registerSCCPPass();
  mlir::affine::registerAffineScalarReplacementPass();
  mlir::registerSymbolDCEPass();
  mlir::registerLocationSnapshotPass();
  mlir::affine::registerAffinePipelineDataTransferPass();

  mlir::affine::registerAffineVectorizePass();
  mlir::affine::registerAffineLoopUnrollPass();
  mlir::affine::registerAffineLoopUnrollAndJamPass();
  mlir::affine::registerSimplifyAffineStructuresPass();
  mlir::affine::registerAffineLoopInvariantCodeMotionPass();
  mlir::affine::registerAffineLoopTilingPass();
  mlir::affine::registerAffineDataCopyGenerationPass();

  mlir::registerConvertAffineToStandardPass();
}

/// Register the interfaces needed to lower to LLVM IR.
void registerLLVMTranslation(mlir::MLIRContext &context);

} // namespace fir::support

#endif // FORTRAN_OPTIMIZER_SUPPORT_INITFIR_H
