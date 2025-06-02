//===--- Function.h - Bytecode function for the VM --------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Function.h"
#include "Program.h"
#include "clang/AST/Decl.h"
#include "clang/AST/DeclCXX.h"

using namespace clang;
using namespace clang::interp;

Function::Function(Program &P, FunctionDeclTy Source, unsigned ArgSize,
                   llvm::SmallVectorImpl<PrimType> &&ParamTypes,
                   llvm::DenseMap<unsigned, ParamDescriptor> &&Params,
                   llvm::SmallVectorImpl<unsigned> &&ParamOffsets,
                   bool HasThisPointer, bool HasRVO, bool IsLambdaStaticInvoker)
    : P(P), Kind(FunctionKind::Normal), Source(Source), ArgSize(ArgSize),
      ParamTypes(std::move(ParamTypes)), Params(std::move(Params)),
      ParamOffsets(std::move(ParamOffsets)), IsValid(false),
      IsFullyCompiled(false), HasThisPointer(HasThisPointer), HasRVO(HasRVO),
      Defined(false) {
  if (const auto *F = dyn_cast<const FunctionDecl *>(Source)) {
    Variadic = F->isVariadic();
    Immediate = F->isImmediateFunction();
    if (const auto *CD = dyn_cast<CXXConstructorDecl>(F)) {
      Virtual = CD->isVirtual();
      Kind = FunctionKind::Ctor;
    } else if (const auto *CD = dyn_cast<CXXDestructorDecl>(F)) {
      Virtual = CD->isVirtual();
      Kind = FunctionKind::Dtor;
    } else if (const auto *MD = dyn_cast<CXXMethodDecl>(F)) {
      Virtual = MD->isVirtual();
      if (IsLambdaStaticInvoker)
        Kind = FunctionKind::LambdaStaticInvoker;
      else if (clang::isLambdaCallOperator(F))
        Kind = FunctionKind::LambdaCallOperator;
      else if (MD->isCopyAssignmentOperator() || MD->isMoveAssignmentOperator())
        Kind = FunctionKind::CopyOrMoveOperator;
    } else {
      Virtual = false;
    }
  } else {
    Variadic = false;
    Virtual = false;
    Immediate = false;
  }
}

Function::ParamDescriptor Function::getParamDescriptor(unsigned Offset) const {
  auto It = Params.find(Offset);
  assert(It != Params.end() && "Invalid parameter offset");
  return It->second;
}

SourceInfo Function::getSource(CodePtr PC) const {
  assert(PC >= getCodeBegin() && "PC does not belong to this function");
  assert(PC <= getCodeEnd() && "PC Does not belong to this function");
  assert(hasBody() && "Function has no body");
  unsigned Offset = PC - getCodeBegin();
  using Elem = std::pair<unsigned, SourceInfo>;
  auto It = llvm::lower_bound(SrcMap, Elem{Offset, {}}, llvm::less_first());
  if (It == SrcMap.end())
    return SrcMap.back().second;
  return It->second;
}
