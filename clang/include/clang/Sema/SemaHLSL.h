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
#include "clang/Basic/DiagnosticSema.h"
#include "clang/Basic/SourceLocation.h"
#include "clang/Sema/SemaBase.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/TargetParser/Triple.h"
#include <initializer_list>

namespace clang {
class AttributeCommonInfo;
class IdentifierInfo;
class InitializedEntity;
class InitializationKind;
class ParsedAttr;
class Scope;
class VarDecl;

namespace hlsl {

// Introduce a wrapper struct around the underlying RootElement. This structure
// will retain extra clang diagnostic information that is not available in llvm.
struct RootSignatureElement {
  RootSignatureElement(SourceLocation Loc,
                       llvm::hlsl::rootsig::RootElement Element)
      : Loc(Loc), Element(Element) {}

  const llvm::hlsl::rootsig::RootElement &getElement() const { return Element; }
  const SourceLocation &getLocation() const { return Loc; }

private:
  SourceLocation Loc;
  llvm::hlsl::rootsig::RootElement Element;
};

} // namespace hlsl

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
// assignments.
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
  HLSLVkConstantIdAttr *
  mergeVkConstantIdAttr(Decl *D, const AttributeCommonInfo &AL, int Id);
  HLSLShaderAttr *mergeShaderAttr(Decl *D, const AttributeCommonInfo &AL,
                                  llvm::Triple::EnvironmentType ShaderType);
  HLSLParamModifierAttr *
  mergeParamModifierAttr(Decl *D, const AttributeCommonInfo &AL,
                         HLSLParamModifierAttr::Spelling Spelling);
  void ActOnTopLevelFunction(FunctionDecl *FD);
  void ActOnVariableDeclarator(VarDecl *VD);
  bool ActOnUninitializedVarDecl(VarDecl *D);
  void ActOnEndOfTranslationUnit(TranslationUnitDecl *TU);
  void CheckEntryPoint(FunctionDecl *FD);
  bool isSemanticValid(FunctionDecl *FD, DeclaratorDecl *D);
  void CheckSemanticAnnotation(FunctionDecl *EntryPoint, const Decl *Param,
                               const HLSLAnnotationAttr *AnnotationAttr);
  void DiagnoseAttrStageMismatch(
      const Attr *A, llvm::Triple::EnvironmentType Stage,
      std::initializer_list<llvm::Triple::EnvironmentType> AllowedStages);

  QualType handleVectorBinOpConversion(ExprResult &LHS, ExprResult &RHS,
                                       QualType LHSType, QualType RHSType,
                                       bool IsCompAssign);
  void emitLogicalOperatorFixIt(Expr *LHS, Expr *RHS, BinaryOperatorKind Opc);

  /// Computes the unique Root Signature identifier from the given signature,
  /// then lookup if there is a previousy created Root Signature decl.
  ///
  /// Returns the identifier and if it was found
  std::pair<IdentifierInfo *, bool>
  ActOnStartRootSignatureDecl(StringRef Signature);

  /// Creates the Root Signature decl of the parsed Root Signature elements
  /// onto the AST and push it onto current Scope
  void
  ActOnFinishRootSignatureDecl(SourceLocation Loc, IdentifierInfo *DeclIdent,
                               ArrayRef<hlsl::RootSignatureElement> Elements);

  void SetRootSignatureOverride(IdentifierInfo *DeclIdent) {
    RootSigOverrideIdent = DeclIdent;
  }

  HLSLRootSignatureDecl *lookupRootSignatureOverrideDecl(DeclContext *DC) const;

  // Returns true if any RootSignatureElement is invalid and a diagnostic was
  // produced
  bool
  handleRootSignatureElements(ArrayRef<hlsl::RootSignatureElement> Elements);
  void handleRootSignatureAttr(Decl *D, const ParsedAttr &AL);
  void handleNumThreadsAttr(Decl *D, const ParsedAttr &AL);
  void handleWaveSizeAttr(Decl *D, const ParsedAttr &AL);
  void handleVkConstantIdAttr(Decl *D, const ParsedAttr &AL);
  void handleVkBindingAttr(Decl *D, const ParsedAttr &AL);
  void handlePackOffsetAttr(Decl *D, const ParsedAttr &AL);
  void handleShaderAttr(Decl *D, const ParsedAttr &AL);
  void handleResourceBindingAttr(Decl *D, const ParsedAttr &AL);
  void handleParamModifierAttr(Decl *D, const ParsedAttr &AL);
  bool handleResourceTypeAttr(QualType T, const ParsedAttr &AL);

  template <typename T>
  T *createSemanticAttr(const ParsedAttr &AL,
                        std::optional<unsigned> Location) {
    T *Attr = ::new (getASTContext()) T(getASTContext(), AL);
    if (Attr->isSemanticIndexable())
      Attr->setSemanticIndex(Location ? *Location : 0);
    else if (Location.has_value()) {
      Diag(Attr->getLocation(), diag::err_hlsl_semantic_indexing_not_supported)
          << Attr->getAttrName()->getName();
      return nullptr;
    }

    return Attr;
  }

  void diagnoseSystemSemanticAttr(Decl *D, const ParsedAttr &AL,
                                  std::optional<unsigned> Index);
  void handleSemanticAttr(Decl *D, const ParsedAttr &AL);

  void handleVkExtBuiltinInputAttr(Decl *D, const ParsedAttr &AL);

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
  bool diagnosePositionType(QualType T, const ParsedAttr &AL);

  bool CanPerformScalarCast(QualType SrcTy, QualType DestTy);
  bool ContainsBitField(QualType BaseTy);
  bool CanPerformElementwiseCast(Expr *Src, QualType DestType);
  bool CanPerformAggregateSplatCast(Expr *Src, QualType DestType);
  ExprResult ActOnOutParamExpr(ParmVarDecl *Param, Expr *Arg);

  QualType getInoutParameterType(QualType Ty);

  bool transformInitList(const InitializedEntity &Entity, InitListExpr *Init);
  bool handleInitialization(VarDecl *VDecl, Expr *&Init);
  void deduceAddressSpace(VarDecl *Decl);

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

  // Global declaration collected for the $Globals default constant
  // buffer which will be created at the end of the translation unit.
  llvm::SmallVector<Decl *> DefaultCBufferDecls;

  uint32_t ImplicitBindingNextOrderID = 0;

  IdentifierInfo *RootSigOverrideIdent = nullptr;

private:
  void collectResourceBindingsOnVarDecl(VarDecl *D);
  void collectResourceBindingsOnUserRecordDecl(const VarDecl *VD,
                                               const RecordType *RT);
  void processExplicitBindingsOnDecl(VarDecl *D);

  void diagnoseAvailabilityViolations(TranslationUnitDecl *TU);

  uint32_t getNextImplicitBindingOrderID() {
    return ImplicitBindingNextOrderID++;
  }

  bool initGlobalResourceDecl(VarDecl *VD);
  bool initGlobalResourceArrayDecl(VarDecl *VD);
  void createResourceRecordCtorArgs(const Type *ResourceTy, StringRef VarName,
                                    HLSLResourceBindingAttr *RBA,
                                    HLSLVkBindingAttr *VkBinding,
                                    uint32_t ArrayIndex,
                                    llvm::SmallVectorImpl<Expr *> &Args);
};

} // namespace clang

#endif // LLVM_CLANG_SEMA_SEMAHLSL_H
