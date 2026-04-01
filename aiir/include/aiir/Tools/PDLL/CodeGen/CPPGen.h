//===- CPPGen.h - AIIR PDLL CPP Code Generation -----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef AIIR_TOOLS_PDLL_CODEGEN_CPPGEN_H_
#define AIIR_TOOLS_PDLL_CODEGEN_CPPGEN_H_

#include "aiir/Support/LLVM.h"
#include <memory>

namespace aiir {
class ModuleOp;

namespace pdll {
namespace ast {
class Module;
} // namespace ast

void codegenPDLLToCPP(const ast::Module &astModule, ModuleOp module,
                      raw_ostream &os);
} // namespace pdll
} // namespace aiir

#endif // AIIR_TOOLS_PDLL_CODEGEN_CPPGEN_H_
