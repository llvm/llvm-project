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
  // TODO(reflection) : support parsing for more reflect-expressions.
  EnterExpressionEvaluationContext Unevaluated(
      Actions, Sema::ExpressionEvaluationContext::Unevaluated);
  assert(Tok.is(tok::caretcaret) && "expecting reflection operator ^^");
  SourceLocation DoubleCaretLoc = ConsumeToken();

  CXXScopeSpec SS;
  if (ParseOptionalCXXScopeSpecifier(SS, /*ObjectType=*/nullptr,
                                     /*ObjectHasErrors=*/false,
                                     /*EnteringContext=*/false)) {
    return ExprError();
  }

  SourceLocation OperandLoc = Tok.getLocation();

  if (Tok.isOneOf(tok::identifier, tok::kw_operator, tok::kw_template,
                  tok::tilde, tok::annot_template_id)) {
    // TODO(reflection) : support parsing for
    // - type-name::
    // - nested-name-specifier identifier ::
    // - namespace-name ::
    // - nested-name-specifier template_opt simple-template-id
    Diag(OperandLoc, diag::err_cannot_reflect_operand);
    return ExprError();
  } else if (SS.isValid() &&
             SS.getScopeRep().getKind() == NestedNameSpecifier::Kind::Global) {
    // global namespace ::.
    Decl *TUDecl = Actions.getASTContext().getTranslationUnitDecl();
    return Actions.ActOnCXXReflectExpr(DoubleCaretLoc, SourceLocation(),
                                       TUDecl);
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
    if (Canon->isBuiltinType()) {
      // Only supports builtin types for now
      return Actions.ActOnCXXReflectExpr(DoubleCaretLoc, TSI);
    }
  }

  Diag(OperandLoc, diag::err_cannot_reflect_operand);
  return ExprError();
}
