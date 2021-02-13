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

#include "mlir/Support/MlirOptMain.h"
#include "flang/Optimizer/Dialect/FIRDialect.h"

using namespace mlir;

int main(int argc, char **argv) {
  fir::registerFIRPasses();
  DialectRegistry registry;
  fir::registerFIRDialects(registry);
  return failed(MlirOptMain(argc, argv, "FIR modular optimizer driver\n",
      registry, /*preloadDialectsInContext*/ false));
}
