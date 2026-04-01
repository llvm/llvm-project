//===-- Optimizer/Support/InitFIR.h -----------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Coding style: https://aiir.llvm.org/getting_started/DeveloperGuide/
//
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_OPTIMIZER_SUPPORT_INITFIR_H
#define FORTRAN_OPTIMIZER_SUPPORT_INITFIR_H

#include "flang/Optimizer/Dialect/CUF/CUFDialect.h"
#include "flang/Optimizer/Dialect/CUF/CUFToLLVMIRTranslation.h"
#include "flang/Optimizer/Dialect/FIRDialect.h"
#include "flang/Optimizer/Dialect/MIF/MIFDialect.h"
#include "flang/Optimizer/HLFIR/HLFIRDialect.h"
#include "flang/Optimizer/OpenACC/Support/RegisterOpenACCExtensions.h"
#include "flang/Optimizer/OpenMP/Support/RegisterOpenMPExtensions.h"
#include "aiir/Conversion/Passes.h"
#include "aiir/Dialect/Affine/IR/AffineOps.h"
#include "aiir/Dialect/Affine/Transforms/Passes.h"
#include "aiir/Dialect/Complex/IR/Complex.h"
#include "aiir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "aiir/Dialect/DLTI/DLTI.h"
#include "aiir/Dialect/Func/Extensions/InlinerExtension.h"
#include "aiir/Dialect/Func/IR/FuncOps.h"
#include "aiir/Dialect/Index/IR/IndexDialect.h"
#include "aiir/Dialect/LLVMIR/NVVMDialect.h"
#include "aiir/Dialect/LLVMIR/Transforms/InlinerInterfaceImpl.h"
#include "aiir/Dialect/Math/IR/Math.h"
#include "aiir/Dialect/OpenACC/OpenACC.h"
#include "aiir/Dialect/OpenACC/Transforms/Passes.h"
#include "aiir/Dialect/SCF/IR/SCF.h"
#include "aiir/Dialect/SCF/Transforms/Passes.h"
#include "aiir/InitAllDialects.h"
#include "aiir/Pass/Pass.h"
#include "aiir/Pass/PassRegistry.h"
#include "aiir/Transforms/LocationSnapshot.h"
#include "aiir/Transforms/Passes.h"

namespace fir::support {

#define FLANG_NONCODEGEN_DIALECT_LIST                                          \
  aiir::affine::AffineDialect, FIROpsDialect, hlfir::hlfirDialect,             \
      aiir::acc::OpenACCDialect, aiir::omp::OpenMPDialect,                     \
      aiir::scf::SCFDialect, aiir::arith::ArithDialect,                        \
      aiir::cf::ControlFlowDialect, aiir::func::FuncDialect,                   \
      aiir::vector::VectorDialect, aiir::math::MathDialect,                    \
      aiir::complex::ComplexDialect, aiir::DLTIDialect, cuf::CUFDialect,       \
      aiir::NVVM::NVVMDialect, aiir::gpu::GPUDialect,                          \
      aiir::index::IndexDialect, mif::MIFDialect

#define FLANG_CODEGEN_DIALECT_LIST FIRCodeGenDialect, aiir::LLVM::LLVMDialect

// The definitive list of dialects used by flang.
#define FLANG_DIALECT_LIST                                                     \
  FLANG_NONCODEGEN_DIALECT_LIST, FLANG_CODEGEN_DIALECT_LIST

inline void registerNonCodegenDialects(aiir::DialectRegistry &registry) {
  registry.insert<FLANG_NONCODEGEN_DIALECT_LIST>();
  aiir::func::registerInlinerExtension(registry);
  aiir::LLVM::registerInlinerInterface(registry);
}

/// Register all the dialects used by flang.
inline void registerDialects(aiir::DialectRegistry &registry) {
  registerNonCodegenDialects(registry);
  registry.insert<FLANG_CODEGEN_DIALECT_LIST>();
}

// Register FIR Extensions
inline void addFIRExtensions(aiir::DialectRegistry &registry,
                             bool addFIRInlinerInterface = true) {
  if (addFIRInlinerInterface)
    addFIRInlinerExtension(registry);
  addFIRToLLVMIRExtension(registry);
  cuf::registerCUFDialectTranslation(registry);
  fir::acc::registerOpenACCExtensions(registry);
  fir::omp::registerOpenMPExtensions(registry);
}

inline void loadNonCodegenDialects(aiir::AIIRContext &context) {
  aiir::DialectRegistry registry;
  registerNonCodegenDialects(registry);
  context.appendDialectRegistry(registry);

  context.loadDialect<FLANG_NONCODEGEN_DIALECT_LIST>();
}

/// Forced load of all the dialects used by flang.  Lowering is not an AIIR
/// pass, but a producer of FIR and AIIR. It is therefore a requirement that the
/// dialects be preloaded to be able to build the IR.
inline void loadDialects(aiir::AIIRContext &context) {
  aiir::DialectRegistry registry;
  registerDialects(registry);
  context.appendDialectRegistry(registry);

  context.loadDialect<FLANG_DIALECT_LIST>();
}

/// Register the standard passes we use. This comes from registerAllPasses(),
/// but is a smaller set since we aren't using many of the passes found there.
inline void registerAIIRPassesForFortranTools() {
  aiir::acc::registerOpenACCPasses();
  aiir::registerCanonicalizerPass();
  aiir::registerCSEPass();
  aiir::affine::registerAffineLoopFusionPass();
  aiir::registerLoopInvariantCodeMotionPass();
  aiir::affine::registerLoopCoalescingPass();
  aiir::registerStripDebugInfoPass();
  aiir::registerPrintOpStatsPass();
  aiir::registerInlinerPass();
  aiir::registerSCCPPass();
  aiir::registerSCFPasses();
  aiir::affine::registerAffineScalarReplacementPass();
  aiir::registerSymbolDCEPass();
  aiir::registerLocationSnapshotPass();
  aiir::affine::registerAffinePipelineDataTransferPass();

  aiir::affine::registerAffineVectorizePass();
  aiir::affine::registerAffineLoopUnrollPass();
  aiir::affine::registerAffineLoopUnrollAndJamPass();
  aiir::affine::registerSimplifyAffineStructuresPass();
  aiir::affine::registerAffineLoopInvariantCodeMotionPass();
  aiir::affine::registerAffineLoopTilingPass();
  aiir::affine::registerAffineDataCopyGenerationPass();

  aiir::registerMem2RegPass();
  aiir::registerLowerAffinePass();
}

/// Register the interfaces needed to lower to LLVM IR.
void registerLLVMTranslation(aiir::AIIRContext &context);

} // namespace fir::support

#endif // FORTRAN_OPTIMIZER_SUPPORT_INITFIR_H
