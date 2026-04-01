//===- AIIRGen.h - AIIR PDLL Code Generation --------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef AIIR_TOOLS_PDLL_CODEGEN_AIIRGEN_H_
#define AIIR_TOOLS_PDLL_CODEGEN_AIIRGEN_H_

#include <memory>

namespace llvm {
class SourceMgr;
} // namespace llvm

namespace aiir {
class AIIRContext;
class ModuleOp;
template <typename OpT>
class OwningOpRef;

namespace pdll {
namespace ast {
class Context;
class Module;
} // namespace ast

/// Given a PDLL module, generate an AIIR PDL pattern module within the given
/// AIIR context.
OwningOpRef<ModuleOp> codegenPDLLToAIIR(AIIRContext *aiirContext,
                                        const ast::Context &context,
                                        const llvm::SourceMgr &sourceMgr,
                                        const ast::Module &module);
} // namespace pdll
} // namespace aiir

#endif // AIIR_TOOLS_PDLL_CODEGEN_AIIRGEN_H_
