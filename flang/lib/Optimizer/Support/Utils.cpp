//===-- Utils.cpp ---------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Coding style: https://mlir.llvm.org/getting_started/DeveloperGuide/
//
//===----------------------------------------------------------------------===//

#include "flang/Optimizer/Support/Utils.h"
#include "flang/Optimizer/Dialect/FIROps.h"

namespace fir {
void buildBindingTables(BindingTables &bindingTables, mlir::ModuleOp mod) {

  // The binding tables are defined in FIR from lowering as fir.dispatch_table
  // operation. Go through each binding tables and store the procedure name and
  // binding index for later use by the fir.dispatch conversion pattern.
  for (auto dispatchTableOp : mod.getOps<fir::DispatchTableOp>()) {
    unsigned bindingIdx = 0;
    BindingTable bindings;
    if (dispatchTableOp.getRegion().empty()) {
      bindingTables[dispatchTableOp.getSymName()] = bindings;
      continue;
    }
    for (auto dtEntry : dispatchTableOp.getBlock().getOps<fir::DTEntryOp>()) {
      bindings[dtEntry.getMethod()] = bindingIdx;
      ++bindingIdx;
    }
    bindingTables[dispatchTableOp.getSymName()] = bindings;
  }
}
} // namespace fir
