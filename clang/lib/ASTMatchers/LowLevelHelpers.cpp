//===- LowLevelHelpers.cpp -------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/ASTMatchers/LowLevelHelpers.h"
#include "clang/AST/Decl.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/Expr.h"
#include "clang/AST/ExprCXX.h"
#include <type_traits>

namespace clang {
namespace ast_matchers {

static const FunctionDecl *getCallee(const CXXConstructExpr &D) {
  return D.getConstructor();
}
static const FunctionDecl *getCallee(const CallExpr &D) {
  return D.getDirectCallee();
}

template <class ExprNode>
static void matchEachArgumentWithParamTypeImpl(
    const ExprNode &Node,
    llvm::function_ref<void(QualType /*Param*/, const Expr * /*Arg*/)>
        OnParamAndArg) {
  static_assert(std::is_same_v<CallExpr, ExprNode> ||
                std::is_same_v<CXXConstructExpr, ExprNode>);
  // The first argument of an overloaded member operator is the implicit object
  // argument of the method which should not be matched against a parameter, so
  // we skip over it here.
  unsigned ArgIndex = 0;
  if (const auto *CE = dyn_cast<CXXOperatorCallExpr>(&Node)) {
    const auto *MD = dyn_cast_or_null<CXXMethodDecl>(CE->getDirectCallee());
    if (MD && !MD->isExplicitObjectMemberFunction()) {
      // This is an overloaded operator call.
      // We need to skip the first argument, which is the implicit object
      // argument of the method which should not be matched against a
      // parameter.
      ++ArgIndex;
    }
  }

  const FunctionProtoType *FProto = nullptr;

  if (const auto *Call = dyn_cast<CallExpr>(&Node)) {
    if (const auto *Value =
            dyn_cast_or_null<ValueDecl>(Call->getCalleeDecl())) {
      QualType QT = Value->getType().getCanonicalType();

      // This does not necessarily lead to a `FunctionProtoType`,
      // e.g. K&R functions do not have a function prototype.
      if (QT->isFunctionPointerType())
        FProto = QT->getPointeeType()->getAs<FunctionProtoType>();

      if (QT->isMemberFunctionPointerType()) {
        const auto *MP = QT->getAs<MemberPointerType>();
        assert(MP && "Must be member-pointer if its a memberfunctionpointer");
        FProto = MP->getPointeeType()->getAs<FunctionProtoType>();
        assert(FProto &&
               "The call must have happened through a member function "
               "pointer");
      }
    }
  }

  unsigned ParamIndex = 0;
  unsigned NumArgs = Node.getNumArgs();
  if (FProto && FProto->isVariadic())
    NumArgs = std::min(NumArgs, FProto->getNumParams());

  for (; ArgIndex < NumArgs; ++ArgIndex, ++ParamIndex) {
    QualType ParamType;
    if (FProto && FProto->getNumParams() > ParamIndex)
      ParamType = FProto->getParamType(ParamIndex);
    else if (const FunctionDecl *FD = getCallee(Node);
             FD && FD->getNumParams() > ParamIndex)
      ParamType = FD->getParamDecl(ParamIndex)->getType();
    else
      continue;

    OnParamAndArg(ParamType, Node.getArg(ArgIndex)->IgnoreParenCasts());
  }
}

void matchEachArgumentWithParamType(
    const CallExpr &Node,
    llvm::function_ref<void(QualType /*Param*/, const Expr * /*Arg*/)>
        OnParamAndArg) {
  matchEachArgumentWithParamTypeImpl(Node, OnParamAndArg);
}

void matchEachArgumentWithParamType(
    const CXXConstructExpr &Node,
    llvm::function_ref<void(QualType /*Param*/, const Expr * /*Arg*/)>
        OnParamAndArg) {
  matchEachArgumentWithParamTypeImpl(Node, OnParamAndArg);
}

} // namespace ast_matchers

} // namespace clang
