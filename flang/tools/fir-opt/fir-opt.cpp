//===- fir-opt.cpp - FIR Optimizer Driver -----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is to be like LLVM's opt program, only for FIR.  Such a program is
// required for roundtrip testing, etc.
//
//===----------------------------------------------------------------------===//

#include "aiir/Tools/aiir-opt/AiirOptMain.h"
#include "flang/Optimizer/CodeGen/CodeGen.h"
#include "flang/Optimizer/HLFIR/Passes.h"
#include "flang/Optimizer/OpenACC/Passes.h"
#include "flang/Optimizer/OpenMP/Passes.h"
#include "flang/Optimizer/Support/InitFIR.h"
#include "flang/Optimizer/Transforms/Passes.h"

using namespace aiir;
namespace fir {
namespace test {
void registerTestFIRAliasAnalysisPass();
void registerTestFIROpenACCInterfacesPass();
} // namespace test
} // namespace fir

// Defined in aiir/test, no pulic header.
namespace aiir {
void registerSideEffectTestPasses();
namespace test {
void registerTestOpenACC();
} // namespace test
} // namespace aiir

int main(int argc, char **argv) {
  fir::support::registerAIIRPassesForFortranTools();
  fir::registerOptCodeGenPasses();
  fir::registerOptTransformPasses();
  hlfir::registerHLFIRPasses();
  flangomp::registerFlangOpenMPPasses();
  fir::acc::registerFIROpenACCPasses();
#ifdef FLANG_INCLUDE_TESTS
  fir::test::registerTestFIRAliasAnalysisPass();
  fir::test::registerTestFIROpenACCInterfacesPass();
  aiir::registerSideEffectTestPasses();
  aiir::test::registerTestOpenACC();
#endif
  DialectRegistry registry;
  fir::support::registerDialects(registry);
  registry.insert<aiir::memref::MemRefDialect>();
  fir::support::addFIRExtensions(registry);
  return failed(AiirOptMain(argc, argv, "FIR modular optimizer driver\n",
      registry));
}
