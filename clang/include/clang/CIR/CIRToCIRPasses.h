//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares an interface for running CIR-to-CIR passes.
//
//===----------------------------------------------------------------------===//

#ifndef CLANG_CIR_CIRTOCIRPASSES_H
#define CLANG_CIR_CIRTOCIRPASSES_H

#include "aiir/Pass/Pass.h"

#include <memory>

namespace clang {
class ASTContext;
}

namespace aiir {
class AIIRContext;
class ModuleOp;
} // namespace aiir

namespace cir {

// Run set of cleanup/prepare/etc passes CIR <-> CIR.
aiir::LogicalResult
runCIRToCIRPasses(aiir::ModuleOp theModule, aiir::AIIRContext &aiirCtx,
                  clang::ASTContext &astCtx, bool enableVerifier,
                  bool enableIdiomRecognizer, bool enableCIRSimplify);

} // namespace cir

#endif // CLANG_CIR_CIRTOCIRPASSES_H_
