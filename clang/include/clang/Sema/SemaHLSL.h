//===----- SemaHLSL.h ----- Semantic Analysis for HLSL constructs ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
/// \file
/// This file declares semantic analysis for HLSL constructs.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_SEMA_SEMAHLSL_H
#define LLVM_CLANG_SEMA_SEMAHLSL_H

#include "clang/AST/ASTFwd.h"
#include "clang/AST/Attr.h"
#include "clang/Basic/SourceLocation.h"
#include "clang/Sema/SemaBase.h"
#include "llvm/TargetParser/Triple.h"
#include <initializer_list>

namespace clang {
class AttributeCommonInfo;
class IdentifierInfo;
class ParsedAttr;
class Scope;

class SemaHLSL : public SemaBase {
public:
  SemaHLSL(Sema &S);

  Decl *ActOnStartBuffer(Scope *BufferScope, bool CBuffer, SourceLocation KwLoc,
                         IdentifierInfo *Ident, SourceLocation IdentLoc,
                         SourceLocation LBrace);
  void ActOnFinishBuffer(Decl *Dcl, SourceLocation RBrace);
  HLSLNumThreadsAttr *mergeNumThreadsAttr(Decl *D,
                                          const AttributeCommonInfo &AL, int X,
                                          int Y, int Z);
  HLSLWaveSizeAttr *mergeWaveSizeAttr(Decl *D, const AttributeCommonInfo &AL,
                                      int Min, int Max, int Preferred,
                                      int SpelledArgsCount);
  HLSLShaderAttr *mergeShaderAttr(Decl *D, const AttributeCommonInfo &AL,
                                  llvm::Triple::EnvironmentType ShaderType);
  HLSLParamModifierAttr *
  mergeParamModifierAttr(Decl *D, const AttributeCommonInfo &AL,
                         HLSLParamModifierAttr::Spelling Spelling);
  void ActOnTopLevelFunction(FunctionDecl *FD);
  void CheckEntryPoint(FunctionDecl *FD);
  void CheckSemanticAnnotation(FunctionDecl *EntryPoint, const Decl *Param,
                               const HLSLAnnotationAttr *AnnotationAttr);
  void DiagnoseAttrStageMismatch(
      const Attr *A, llvm::Triple::EnvironmentType Stage,
      std::initializer_list<llvm::Triple::EnvironmentType> AllowedStages);
  void DiagnoseAvailabilityViolations(TranslationUnitDecl *TU);

  void handleNumThreadsAttr(Decl *D, const ParsedAttr &AL);
  void handleWaveSizeAttr(Decl *D, const ParsedAttr &AL);
  void handleSV_DispatchThreadIDAttr(Decl *D, const ParsedAttr &AL);
  void handlePackOffsetAttr(Decl *D, const ParsedAttr &AL);
  void handleShaderAttr(Decl *D, const ParsedAttr &AL);
  void handleROVAttr(Decl *D, const ParsedAttr &AL);
  void handleResourceClassAttr(Decl *D, const ParsedAttr &AL);
  void handleResourceBindingAttr(Decl *D, const ParsedAttr &AL);
  void handleParamModifierAttr(Decl *D, const ParsedAttr &AL);
  bool handleResourceTypeAttr(const ParsedAttr &AL);

  bool CheckBuiltinFunctionCall(unsigned BuiltinID, CallExpr *TheCall);
  QualType ProcessResourceTypeAttributes(QualType Wrapped);
  SourceLocation TakeLocForHLSLAttribute(const HLSLAttributedResourceType *RT);

  // HLSL Type trait implementations
  bool IsScalarizedLayoutCompatible(QualType T1, QualType T2) const;

  bool CheckCompatibleParameterABI(FunctionDecl *New, FunctionDecl *Old);

  ExprResult ActOnOutParamExpr(ParmVarDecl *Param, Expr *Arg);

  QualType getInoutParameterType(QualType Ty);
};

} // namespace clang

#endif // LLVM_CLANG_SEMA_SEMAHLSL_H
