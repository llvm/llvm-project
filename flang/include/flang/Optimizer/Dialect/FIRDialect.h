//===-- Optimizer/Dialect/FIRDialect.h -- FIR dialect -----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef OPTIMIZER_DIALECT_FIRDIALECT_H
#define OPTIMIZER_DIALECT_FIRDIALECT_H

#include "mlir/IR/Dialect.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"

namespace fir {

/// FIR dialect
class FIROpsDialect final : public mlir::Dialect {
public:
  explicit FIROpsDialect(mlir::MLIRContext *ctx);
  virtual ~FIROpsDialect();

  static llvm::StringRef getDialectNamespace() { return "fir"; }

  mlir::Type parseType(mlir::DialectAsmParser &parser) const override;
  void printType(mlir::Type ty, mlir::DialectAsmPrinter &p) const override;

  mlir::Attribute parseAttribute(mlir::DialectAsmParser &parser,
                                 mlir::Type type) const override;
  void printAttribute(mlir::Attribute attr,
                      mlir::DialectAsmPrinter &p) const override;
};

/// Register the dialect with MLIR
inline void registerFIR() {
  // we want to register exactly once
  [[maybe_unused]] static bool init_once = [] {
    mlir::registerDialect<mlir::AffineDialect>();
    mlir::registerDialect<mlir::LLVM::LLVMDialect>();
    mlir::registerDialect<mlir::omp::OpenMPDialect>();
    mlir::registerDialect<mlir::scf::SCFDialect>();
    mlir::registerDialect<mlir::StandardOpsDialect>();
    mlir::registerDialect<mlir::vector::VectorDialect>();
    mlir::registerDialect<FIROpsDialect>();
    return true;
  }();
}

/// Register the standard passes we use. This comes from registerAllPasses(),
/// but is a smaller set since we aren't using many of the passes found there.
inline void registerGeneralPasses() {
  mlir::createCanonicalizerPass();
  mlir::createCSEPass();
  mlir::createSuperVectorizePass({});
  mlir::createLoopUnrollPass();
  mlir::createLoopUnrollAndJamPass();
  mlir::createSimplifyAffineStructuresPass();
  mlir::createLoopFusionPass();
  mlir::createLoopInvariantCodeMotionPass();
  mlir::createAffineLoopInvariantCodeMotionPass();
  mlir::createPipelineDataTransferPass();
  mlir::createLowerAffinePass();
  mlir::createLoopTilingPass(0);
  mlir::createLoopCoalescingPass();
  mlir::createAffineDataCopyGenerationPass(0, 0);
  mlir::createMemRefDataFlowOptPass();
  mlir::createStripDebugInfoPass();
  mlir::createPrintOpStatsPass();
  mlir::createInlinerPass();
  mlir::createSymbolDCEPass();
  mlir::createLocationSnapshotPass({});
}

inline void registerFIRPasses() { registerGeneralPasses(); }

} // namespace fir

#endif // OPTIMIZER_DIALECT_FIRDIALECT_H
