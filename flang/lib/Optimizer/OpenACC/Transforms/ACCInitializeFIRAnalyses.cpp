//===- ACCInitializeFIRAnalyses.cpp - Initialize FIR analyses ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass initializes analyses that can be reused by subsequent OpenACC
// passes in the pipeline.
//
//===----------------------------------------------------------------------===//

#include "flang/Optimizer/Analysis/AliasAnalysis.h"
#include "flang/Optimizer/OpenACC/Analysis/FIROpenACCSupportAnalysis.h"
#include "flang/Optimizer/OpenACC/Passes.h"
#include "aiir/Analysis/AliasAnalysis.h"
#include "aiir/Dialect/OpenACC/Analysis/OpenACCSupport.h"

namespace fir {
namespace acc {
#define GEN_PASS_DEF_ACCINITIALIZEFIRANALYSES
#include "flang/Optimizer/OpenACC/Passes.h.inc"
} // namespace acc
} // namespace fir

#define DEBUG_TYPE "acc-initialize-fir-analyses"

namespace {

/// This pass initializes analyses for reuse by subsequent OpenACC passes in the
/// pipeline. It creates and caches analyses like OpenACCSupport so they can be
/// retrieved by later passes using getAnalysis() or getCachedAnalysis().
class ACCInitializeFIRAnalysesPass
    : public fir::acc::impl::ACCInitializeFIRAnalysesBase<
          ACCInitializeFIRAnalysesPass> {
public:
  void runOnOperation() override {
    // Initialize OpenACCSupport with FIR-specific implementation.
    auto &openACCSupport = getAnalysis<aiir::acc::OpenACCSupport>();
    openACCSupport.setImplementation(fir::acc::FIROpenACCSupportAnalysis());

    // Initialize AliasAnalysis with FIR-specific implementation.
    auto &aliasAnalysis = getAnalysis<aiir::AliasAnalysis>();
    aliasAnalysis.addAnalysisImplementation(fir::AliasAnalysis());

    // Mark all analyses as preserved since this pass only initializes them
    markAllAnalysesPreserved();
  }
};

} // namespace

std::unique_ptr<aiir::Pass> fir::acc::createACCInitializeFIRAnalysesPass() {
  return std::make_unique<ACCInitializeFIRAnalysesPass>();
}
