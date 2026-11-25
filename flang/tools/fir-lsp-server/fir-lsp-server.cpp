//===-- tools/fir-lsp-server/fir-lsp-server.cpp ---------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Tools/mlir-lsp-server/MlirLspServerMain.h"
#include "flang/Optimizer/Support/InitFIR.h"

int main(int argc, char **argv) {
  mlir::DialectRegistry registry;
  fir::support::registerNonCodegenDialects(registry);
  fir::support::addFIRExtensions(registry);
  return mlir::failed(mlir::MlirLspServerMain(argc, argv, registry));
}
