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
#include "clang/Basic/Builtins.h"

using namespace clang;
using namespace clang::interp;

Function::Function(Program &P, FunctionDeclTy Source, unsigned ArgSize,
                   llvm::SmallVectorImpl<PrimType> &&ParamTypes,
                   llvm::DenseMap<unsigned, ParamDescriptor> &&Params,
                   llvm::SmallVectorImpl<unsigned> &&ParamOffsets,
                   bool HasThisPointer, bool HasRVO)
    : P(P), Kind(FunctionKind::Normal), Source(Source), ArgSize(ArgSize),
      ParamTypes(std::move(ParamTypes)), Params(std::move(Params)),
      ParamOffsets(std::move(ParamOffsets)), HasThisPointer(HasThisPointer),
      HasRVO(HasRVO) {
  if (const auto *F = dyn_cast<const FunctionDecl *>(Source)) {
    Variadic = F->isVariadic();
    BuiltinID = F->getBuiltinID();
    if (const auto *CD = dyn_cast<CXXConstructorDecl>(F)) {
      Virtual = CD->isVirtual();
      Kind = FunctionKind::Ctor;
    } else if (const auto *CD = dyn_cast<CXXDestructorDecl>(F)) {
      Virtual = CD->isVirtual();
      Kind = FunctionKind::Dtor;
    } else if (const auto *MD = dyn_cast<CXXMethodDecl>(F)) {
      Virtual = MD->isVirtual();
      if (MD->isLambdaStaticInvoker())
        Kind = FunctionKind::LambdaStaticInvoker;
      else if (clang::isLambdaCallOperator(F))
        Kind = FunctionKind::LambdaCallOperator;
    }
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

/// Unevaluated builtins don't get their arguments put on the stack
/// automatically. They instead operate on the AST of their Call
/// Expression.
/// Similar information is available via ASTContext::BuiltinInfo,
/// but that is not correct for our use cases.
static bool isUnevaluatedBuiltin(unsigned BuiltinID) {
  return BuiltinID == Builtin::BI__builtin_classify_type ||
         BuiltinID == Builtin::BI__builtin_os_log_format_buffer_size ||
         BuiltinID == Builtin::BI__builtin_constant_p ||
         BuiltinID == Builtin::BI__noop;
}

bool Function::isUnevaluatedBuiltin() const {
  return ::isUnevaluatedBuiltin(BuiltinID);
}
