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

ExprResult Parser::ParseCXXReflectExpression() {
  // TODO(reflection) : support parsing for global namespace,
  // reflection-name, id-expression and remaining supports for
  // type-id (placeholder type, alias template, etc.)
  EnterExpressionEvaluationContext Unevaluated(
      Actions, Sema::ExpressionEvaluationContext::Unevaluated);
  assert(Tok.is(tok::caretcaret));
  SourceLocation CaretCaretLoc = ConsumeToken();
  SourceLocation OperandLoc = Tok.getLocation();

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

    QT = QT.getCanonicalType().getUnqualifiedType();
    if (TSI && QT.getTypePtr()->isBuiltinType()) {
      // Only supports builtin types for now
      return Actions.ActOnCXXReflectExpr(CaretCaretLoc, TSI);
    }
  }

  Diag(OperandLoc, diag::err_cannot_reflect_operand);
  return ExprError();
}
