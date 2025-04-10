//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the main function for cir-link, a tool to link CIR
// modules: cir-link a.mlir b.mlir c.mlir -o x.mlir
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/DLTI/DLTI.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Tools/mlir-link/MlirLinkMain.h"
#include "mlir/IR/BuiltinLinkerInterface.h"
#include "clang/CIR/Dialect/IR/CIRDialect.h"
#include "clang/CIR/Interfaces/CIRLinkerInterface.h"

using namespace mlir;

int main(int argc, char **argv) {
  DialectRegistry registry;
  registry.insert<mlir::BuiltinDialect, cir::CIRDialect, mlir::DLTIDialect>();
  builtin::registerLinkerInterface(registry);
  cir::registerLinkerInterface(registry);

  return failed(MlirLinkMain(argc, argv, registry));
}
