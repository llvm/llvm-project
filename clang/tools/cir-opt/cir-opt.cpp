//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Similar to AIIR/LLVM's "opt" tools but also deals with analysis and custom
// arguments. TODO: this is basically a copy from AiirOptMain.cpp, but capable
// of module emission as specified by the user.
//
//===----------------------------------------------------------------------===//

#include "aiir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"
#include "aiir/Dialect/DLTI/DLTI.h"
#include "aiir/Dialect/Func/IR/FuncOps.h"
#include "aiir/Dialect/LLVMIR/LLVMDialect.h"
#include "aiir/Dialect/MemRef/IR/MemRef.h"
#include "aiir/Dialect/OpenMP/OpenMPDialect.h"
#include "aiir/IR/BuiltinDialect.h"
#include "aiir/Pass/PassManager.h"
#include "aiir/Pass/PassOptions.h"
#include "aiir/Pass/PassRegistry.h"
#include "aiir/Tools/aiir-opt/AiirOptMain.h"
#include "aiir/Transforms/Passes.h"
#include "clang/CIR/Dialect/IR/CIRDialect.h"
#include "clang/CIR/Dialect/Passes.h"
#include "clang/CIR/Passes.h"

struct CIRToLLVMPipelineOptions
    : public aiir::PassPipelineOptions<CIRToLLVMPipelineOptions> {};

int main(int argc, char **argv) {
  // TODO: register needed AIIR passes for CIR?
  aiir::DialectRegistry registry;
  registry.insert<aiir::BuiltinDialect, cir::CIRDialect,
                  aiir::memref::MemRefDialect, aiir::LLVM::LLVMDialect,
                  aiir::DLTIDialect, aiir::omp::OpenMPDialect>();

  ::aiir::registerPass([]() -> std::unique_ptr<::aiir::Pass> {
    return aiir::createCIRCanonicalizePass();
  });
  ::aiir::registerPass([]() -> std::unique_ptr<::aiir::Pass> {
    return aiir::createCIRSimplifyPass();
  });

  aiir::PassPipelineRegistration<CIRToLLVMPipelineOptions> pipeline(
      "cir-to-llvm", "",
      [](aiir::OpPassManager &pm, const CIRToLLVMPipelineOptions &options) {
        cir::direct::populateCIRToLLVMPasses(pm);
      });

  ::aiir::registerPass([]() -> std::unique_ptr<::aiir::Pass> {
    return aiir::createCIRFlattenCFGPass();
  });

  ::aiir::registerPass([]() -> std::unique_ptr<::aiir::Pass> {
    return aiir::createCIREHABILoweringPass();
  });

  ::aiir::registerPass([]() -> std::unique_ptr<::aiir::Pass> {
    return aiir::createHoistAllocasPass();
  });

  ::aiir::registerPass([]() -> std::unique_ptr<::aiir::Pass> {
    return aiir::createGotoSolverPass();
  });

  aiir::registerTransformsPasses();

  return aiir::asMainReturnCode(AiirOptMain(
      argc, argv, "Clang IR analysis and optimization tool\n", registry));
}
