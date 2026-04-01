//===- AIIRGen.h - AIIR Generation from a Toy AST -------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares a simple interface to perform IR generation targeting AIIR
// from a Module AST for the Toy language.
//
//===----------------------------------------------------------------------===//

#ifndef TOY_AIIRGEN_H
#define TOY_AIIRGEN_H

#include <memory>

namespace aiir {
class AIIRContext;
template <typename OpTy>
class OwningOpRef;
class ModuleOp;
} // namespace aiir

namespace toy {
class ModuleAST;

/// Emit IR for the given Toy moduleAST, returns a newly created AIIR module
/// or nullptr on failure.
aiir::OwningOpRef<aiir::ModuleOp> aiirGen(aiir::AIIRContext &context,
                                          ModuleAST &moduleAST);
} // namespace toy

#endif // TOY_AIIRGEN_H
