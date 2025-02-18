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
#include "clang/AST/Type.h"
#include "clang/AST/TypeLoc.h"
#include "clang/Basic/SourceLocation.h"
#include "clang/Sema/SemaBase.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/TargetParser/Triple.h"
#include <initializer_list>

namespace clang {
class AttributeCommonInfo;
class IdentifierInfo;
class ParsedAttr;
class Scope;
class VarDecl;

using llvm::dxil::ResourceClass;

// FIXME: This can be hidden (as static function in SemaHLSL.cpp) once we no
// longer need to create builtin buffer types in HLSLExternalSemaSource.
bool CreateHLSLAttributedResourceType(
    Sema &S, QualType Wrapped, ArrayRef<const Attr *> AttrList,
    QualType &ResType, HLSLAttributedResourceLocInfo *LocInfo = nullptr);

enum class BindingType : uint8_t { NotAssigned, Explicit, Implicit };

// DeclBindingInfo struct stores information about required/assigned resource
// binding onon a declaration for specific resource class.
struct DeclBindingInfo {
  const VarDecl *Decl;
  ResourceClass ResClass;
  const HLSLResourceBindingAttr *Attr;
  BindingType BindType;

  DeclBindingInfo(const VarDecl *Decl, ResourceClass ResClass,
                  BindingType BindType = BindingType::NotAssigned,
                  const HLSLResourceBindingAttr *Attr = nullptr)
      : Decl(Decl), ResClass(ResClass), Attr(Attr), BindType(BindType) {}

  void setBindingAttribute(HLSLResourceBindingAttr *A, BindingType BT) {
    assert(Attr == nullptr && BindType == BindingType::NotAssigned &&
           "binding attribute already assigned");
    Attr = A;
    BindType = BT;
  }
};

// ResourceBindings class stores information about all resource bindings
// in a shader. It is used for binding diagnostics and implicit binding
// assigments.
class ResourceBindings {
public:
  DeclBindingInfo *addDeclBindingInfo(const VarDecl *VD,
                                      ResourceClass ResClass);
  DeclBindingInfo *getDeclBindingInfo(const VarDecl *VD,
                                      ResourceClass ResClass);
  bool hasBindingInfoForDecl(const VarDecl *VD) const;

private:
  // List of all resource bindings required by the shader.
  // A global declaration can have multiple bindings for different
  // resource classes. They are all stored sequentially in this list.
  // The DeclToBindingListIndex hashtable maps a declaration to the
  // index of the first binding info in the list.
  llvm::SmallVector<DeclBindingInfo> BindingsList;
  llvm::DenseMap<const VarDecl *, unsigned> DeclToBindingListIndex;
};

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
  void ActOnVariableDeclarator(VarDecl *VD);
  void CheckEntryPoint(FunctionDecl *FD);
  void CheckSemanticAnnotation(FunctionDecl *EntryPoint, const Decl *Param,
                               const HLSLAnnotationAttr *AnnotationAttr);
  void DiagnoseAttrStageMismatch(
      const Attr *A, llvm::Triple::EnvironmentType Stage,
      std::initializer_list<llvm::Triple::EnvironmentType> AllowedStages);
  void DiagnoseAvailabilityViolations(TranslationUnitDecl *TU);

  QualType handleVectorBinOpConversion(ExprResult &LHS, ExprResult &RHS,
                                       QualType LHSType, QualType RHSType,
                                       bool IsCompAssign);
  void emitLogicalOperatorFixIt(Expr *LHS, Expr *RHS, BinaryOperatorKind Opc);

  void handleNumThreadsAttr(Decl *D, const ParsedAttr &AL);
  void handleWaveSizeAttr(Decl *D, const ParsedAttr &AL);
  void handleSV_DispatchThreadIDAttr(Decl *D, const ParsedAttr &AL);
  void handleSV_GroupThreadIDAttr(Decl *D, const ParsedAttr &AL);
  void handleSV_GroupIDAttr(Decl *D, const ParsedAttr &AL);
  void handlePackOffsetAttr(Decl *D, const ParsedAttr &AL);
  void handleShaderAttr(Decl *D, const ParsedAttr &AL);
  void handleResourceBindingAttr(Decl *D, const ParsedAttr &AL);
  void handleParamModifierAttr(Decl *D, const ParsedAttr &AL);
  bool handleResourceTypeAttr(QualType T, const ParsedAttr &AL);

  bool CheckBuiltinFunctionCall(unsigned BuiltinID, CallExpr *TheCall);
  QualType ProcessResourceTypeAttributes(QualType Wrapped);
  HLSLAttributedResourceLocInfo
  TakeLocForHLSLAttribute(const HLSLAttributedResourceType *RT);

  // HLSL Type trait implementations
  bool IsScalarizedLayoutCompatible(QualType T1, QualType T2) const;
  bool IsTypedResourceElementCompatible(QualType T1);

  bool CheckCompatibleParameterABI(FunctionDecl *New, FunctionDecl *Old);

  // Diagnose whether the input ID is uint/unit2/uint3 type.
  bool diagnoseInputIDType(QualType T, const ParsedAttr &AL);

  ExprResult ActOnOutParamExpr(ParmVarDecl *Param, Expr *Arg);

  QualType getInoutParameterType(QualType Ty);

private:
  // HLSL resource type attributes need to be processed all at once.
  // This is a list to collect them.
  llvm::SmallVector<const Attr *> HLSLResourcesTypeAttrs;

  /// TypeLoc data for HLSLAttributedResourceType instances that we
  /// have not yet populated.
  llvm::DenseMap<const HLSLAttributedResourceType *,
                 HLSLAttributedResourceLocInfo>
      LocsForHLSLAttributedResources;

  // List of all resource bindings
  ResourceBindings Bindings;

private:
  void collectResourcesOnVarDecl(VarDecl *D);
  void collectResourcesOnUserRecordDecl(const VarDecl *VD,
                                        const RecordType *RT);
  void processExplicitBindingsOnDecl(VarDecl *D);
};

} // namespace clang

#endif // LLVM_CLANG_SEMA_SEMAHLSL_H
