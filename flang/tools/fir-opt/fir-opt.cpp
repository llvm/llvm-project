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

#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "flang/Optimizer/CodeGen/CodeGen.h"
#include "flang/Optimizer/Support/InitFIR.h"
#include "flang/Optimizer/Transforms/Passes.h"

using namespace mlir;

int main(int argc, char **argv) {
  fir::support::registerMLIRPassesForFortranTools();
  fir::registerOptCodeGenPasses();
  fir::registerOptTransformPasses();
  DialectRegistry registry;
  fir::support::registerDialects(registry);
  return failed(MlirOptMain(argc, argv, "FIR modular optimizer driver\n",
      registry, /*preloadDialectsInContext=*/false));
}
