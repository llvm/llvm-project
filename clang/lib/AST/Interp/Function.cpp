//===--- Function.h - Bytecode function for the VM --------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Function.h"
#include "Program.h"
#include "Opcode.h"
#include "clang/AST/Decl.h"
#include "clang/AST/DeclCXX.h"

using namespace clang;
using namespace clang::interp;

Function::Function(Program &P, const FunctionDecl *F, unsigned ArgSize,
                   llvm::SmallVector<PrimType, 8> &&ParamTypes,
                   llvm::DenseMap<unsigned, ParamDescriptor> &&Params)
    : P(P), Loc(F->getBeginLoc()), F(F), ArgSize(ArgSize),
      ParamTypes(std::move(ParamTypes)), Params(std::move(Params)) {}

CodePtr Function::getCodeBegin() const { return Code.data(); }

CodePtr Function::getCodeEnd() const { return Code.data() + Code.size(); }

Function::ParamDescriptor Function::getParamDescriptor(unsigned Offset) const {
  auto It = Params.find(Offset);
  assert(It != Params.end() && "Invalid parameter offset");
  return It->second;
}

SourceInfo Function::getSource(CodePtr PC) const {
  unsigned Offset = PC - getCodeBegin();
  using Elem = std::pair<unsigned, SourceInfo>;
  auto It = std::lower_bound(SrcMap.begin(), SrcMap.end(), Elem{Offset, {}},
                             [](Elem A, Elem B) { return A.first < B.first; });
  if (It == SrcMap.end() || It->first != Offset)
    llvm::report_fatal_error("missing source location");
  return It->second;
}

bool Function::isVirtual() const {
  if (auto *M = dyn_cast<CXXMethodDecl>(F))
    return M->isVirtual();
  return false;
}
