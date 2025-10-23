//===--- ParseReflect.cpp - C++26 Reflection Parsing ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//  This file implements parsing for reflection facilities.
//
//===----------------------------------------------------------------------===//

#include "clang/AST/LocInfoType.h"
#include "clang/Basic/DiagnosticParse.h"
#include "clang/Parse/Parser.h"
#include "clang/Sema/EnterExpressionEvaluationContext.h"
using namespace clang;

ExprResult Parser::ParseCXXReflectExpression(SourceLocation OpLoc) {
  // TODO(reflection) : support parsing for more reflect-expressions.
  EnterExpressionEvaluationContext Unevaluated(
      Actions, Sema::ExpressionEvaluationContext::Unevaluated);

  SourceLocation OperandLoc = Tok.getLocation();

  {
    TentativeParsingAction TPA(*this);
    // global namespace ::
    if (Tok.is(tok::coloncolon)) {
      ConsumeToken();
      TPA.Commit();
      Decl *TUDecl = Actions.getASTContext().getTranslationUnitDecl();
      return Actions.ActOnCXXReflectExpr(OpLoc, SourceLocation(), TUDecl);
    }
    TPA.Revert();
  }

  if (isCXXTypeId(TentativeCXXTypeIdContext::AsReflectionOperand)) {
    TypeResult TR = ParseTypeName(/*TypeOf=*/nullptr);
    if (TR.isInvalid())
      return ExprError();

    TypeSourceInfo *TSI = nullptr;
    QualType QT = Actions.GetTypeFromParser(TR.get(), &TSI);

    if (QT.isNull())
      return ExprError();

    if (!TSI)
      TSI = Actions.getASTContext().getTrivialTypeSourceInfo(QT, OperandLoc);

    QualType Canon = QT.getCanonicalType();
    if(Canon->isBuiltinType()) {
      // Only supports builtin types for now
      return Actions.ActOnCXXReflectExpr(OpLoc, TSI);
    }
  }

  Diag(OperandLoc, diag::err_cannot_reflect_operand);
  return ExprError();
}
