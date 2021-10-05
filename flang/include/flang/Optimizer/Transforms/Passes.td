//===-- Passes.td - Transforms pass definition file --------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains definitions for passes within the Optimizer/Transforms/
//  directory.
//
//===----------------------------------------------------------------------===//

#ifndef FLANG_OPTIMIZER_TRANSFORMS_PASSES
#define FLANG_OPTIMIZER_TRANSFORMS_PASSES

include "mlir/Pass/PassBase.td"

def ExternalNameConversion : Pass<"external-name-interop", "mlir::ModuleOp"> {
  let summary = "Convert name for external interoperability";
  let description = [{
    Demangle FIR internal name and mangle them for external interoperability.
  }];
  let constructor = "::fir::createExternalNameConversionPass()";
}

#endif // FLANG_OPTIMIZER_TRANSFORMS_PASSES
