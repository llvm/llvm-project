//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements machinery for any CIR <-> CIR passes used by clang.
//
//===----------------------------------------------------------------------===//

// #include "clang/AST/ASTContext.h"
#include "clang/CIR/Dialect/Passes.h"

#include "mlir/Pass/PassManager.h"

namespace mlir {

void populateCIRPreLoweringPasses(OpPassManager &pm) {
  pm.addPass(createCIRFlattenCFGPass());
}

} // namespace mlir
