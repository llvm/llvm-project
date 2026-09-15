//===--- ErrorRecovery.h - Errory Recovery Impl --------------*- C++
//-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_INTERPRETER_ERROR_RECOVERY_H
#define LLVM_CLANG_INTERPRETER_ERROR_RECOVERY_H

#include "clang/AST/ASTContext.h"
#include "clang/AST/ASTMutationListener.h"
#include "clang/AST/Decl.h"
#include "clang/AST/DeclBase.h"
#include "clang/Sema/Sema.h"

#include "llvm/ADT/DenseMap.h"

namespace clang {
class ASTContext;

struct SemaStashCheckPoint {
  /// Sema::*
  llvm::SlabCheckPoint SemaBumpSlabCP;
  size_t CachedFunctionScopeSize = 0;
  size_t FunctionScopesSize = 0;
  size_t Ident_superSize = 0;
  size_t PragmaClangBSSSectionSize = 0;
  size_t PragmaClangDataSectionSize = 0;
  size_t PragmaClangRodataSectionSize = 0;
  size_t PragmaClangRelroSectionSize = 0;
  size_t PragmaClangTextSectionSize = 0;
  size_t VtorDispStackSize = 0;
  size_t AlignPackStackSize = 0;
  size_t AlignPackIncludeStackSize = 0;
  size_t DataSegStackSize = 0;
  size_t BSSSegStackSize = 0;
  size_t ConstSegStackSize = 0;
  size_t CodeSegStackSize = 0;
  size_t StrictGuardStackCheckStackSize = 0;
  size_t FpPragmaStackSize = 0;
  size_t FunctionToSectionMapSize = 0;
  size_t PragmaAttributeStackSize = 0;
  size_t MSFunctionNoBuiltinsSize = 0;
  size_t PendingExportedNamesSize = 0;
  size_t TypoCorrectedFunctionDefinitionsSize = 0;
  size_t FlagBitsCacheSize = 0;
  size_t AssignEnumCacheSize = 0;
  size_t WeakUndeclaredIdentifiersSize = 0;
  size_t ExtnameUndeclaredIdentifiersSize = 0;
  size_t UnusedLocalTypedefNameCandidatesSize = 0;
  Sema::UnusedFileScopedDeclsType::iterator UnusedFileScopedDeclsSize;
  Sema::TentativeDefinitionsType::iterator TentativeDefinitionsSize;
  size_t ExternalDeclarationsSize = 0;
  size_t ParsingInitForAutoVarsSize = 0;
  size_t DeclsToCheckForDeferredDiagsSize = 0;
  size_t ShadowingDeclsSize = 0;
  size_t WeakTopLevelDeclSize = 0;
  Sema::ExtVectorDeclsType::iterator ExtVectorDeclsSize;
  size_t VTableUsesSize = 0;
  size_t VTablesUsedSize = 0;
  size_t DelayedDllExportClassesSize = 0;
  size_t DelayedDllExportMemberFunctionsSize = 0;
  size_t InventedParameterInfosSize = 0;
  size_t FieldCollectorSize = 0;
  size_t UnusedPrivateFieldsSize = 0;
  size_t PureVirtualClassDiagSetSize = 0;
  Sema::DelegatingCtorDeclsType::iterator DelegatingCtorDeclsSize;
  size_t StdNamespaceSize = 0;
  size_t UnparsedDefaultArgLocsSize = 0;
  size_t UndefinedButUsedSize = 0;
  size_t SpecialMembersBeingDeclaredSize = 0;
  size_t DelayedOverridingExceptionSpecChecksSize = 0;
  size_t DelayedEquivalentExceptionSpecChecksSize = 0;
  size_t MaybeODRUseExprsSize = 0;
  size_t RefsMinusAssignmentsSize = 0;
  size_t ExprCleanupObjectsSize = 0;
  size_t ExprEvalContextsSize = 0;
  size_t FailedImmediateInvocationsSize = 0;
  size_t ImplicitlyRetainedSelfLocsSize = 0;
  size_t DeleteExprsSize = 0;
  size_t CurrentParameterCopyTypesSize = 0;
  size_t AggregateDeductionCandidatesSize = 0;
  size_t TypoCorrectionFailuresSize = 0;
  size_t SpecialMemberCacheSize = 0;
  size_t ModuleScopesSize = 0;
  size_t DeferredExportedNamespacesSize = 0;
  size_t PendingInlineFuncDeclsSize = 0;
  size_t CurrentSEHFinallySize = 0;
  size_t CurrentDeferSize = 0;
  size_t LateParsedTemplateMapSize = 0;
  size_t SuppressedDiagnosticsSize = 0;
  size_t CurrentInstantiationScopeSize = 0;
  size_t UnparsedDefaultArgInstantiationsSize = 0;
  size_t CodeSynthesisContextsSize = 0;
  size_t InstantiatingSpecializationsSize = 0;
  size_t InstantiatedNonDependentTypesSize = 0;
  size_t CodeSynthesisContextLookupModulesSize = 0;
  size_t LookupModulesCacheSize = 0;
  size_t VisibleNamespaceCacheSize = 0;
  // size_t TemplateInstCallbacksSize = 0;
  size_t PendingInstantiationsSize = 0;
  size_t LateParsedInstantiationsSize = 0;
  size_t SavedVTableUsesSize = 0;
  size_t SavedPendingInstantiationsSize = 0;
  size_t PendingLocalImplicitInstantiationsSize = 0;
  size_t CurrentCachedTemplateArgsSize = 0;
  size_t UnsubstitutedConstraintSatisfactionCacheSize = 0;
  size_t SubsumptionCacheSize = 0;
  size_t NormalizationCacheSize = 0;
  size_t SatisfactionCacheSize = 0;
  size_t SatisfactionStackSize = 0;
  //   size_t NullabilityMapSize = 0;
  size_t DeclsWithEffectsToVerifySize = 0;
  //   size_t AllEffectsToVerifySize = 0;
};

/// Stashes and restores persistent Sema state around an incremental parse.
///
/// Usage:
///   SemaStashCheckPoint CP;
///   SemaStateRecovery Stash(S);
///   Stash.stash(CP);
///   // ... parse ...
///   if (failed)
///     Stash.restore(CP, ASTSlabCP);
class SemaStateRecovery {
  Sema &S;

  /// Holds full copies of PragmaStack, PragmaClangSection, FileNullabilityMap,
  /// etc.
  //   struct PragmaSnapshot;
  //   std::unique_ptr<PragmaSnapshot> Pragmas;

public:
  explicit SemaStateRecovery(Sema &S) : S(S) {}
  //   ~SemaStateRecovery();

  void stash(SemaStashCheckPoint &CP);
  void restore(SemaStashCheckPoint &CP, llvm::SlabCheckPoint SlabCP);
};

struct StashCheckPoint {
  // Types vector
  size_t TypesSize = 0;

  // FoldingSets — store count of nodes
  size_t ExtQualNodesSize = 0;
  size_t ComplexTypesSize = 0;
  size_t PointerTypesSize = 0;
  size_t AdjustedTypesSize = 0;
  size_t BlockPointerTypesSize = 0;
  size_t LValueReferenceTypesSize = 0;
  size_t RValueReferenceTypesSize = 0;
  size_t MemberPointerTypesSize = 0;

  size_t ConstantArrayTypesSize = 0;
  size_t IncompleteArrayTypesSize = 0;
  size_t VariableArrayTypesSize = 0;

  size_t DependentSizedArrayTypesSize = 0;
  size_t DependentSizedExtVectorTypesSize = 0;
  size_t DependentAddressSpaceTypesSize = 0;
  size_t VectorTypesSize = 0;
  size_t DependentVectorTypesSize = 0;
  size_t MatrixTypesSize = 0;
  size_t DependentSizedMatrixTypesSize = 0;
  size_t FunctionNoProtoTypesSize = 0;
  size_t FunctionProtoTypesSize = 0;
  size_t DependentTypeOfExprTypesSize = 0;
  size_t DependentDecltypeTypesSize = 0;

  size_t DependentPackIndexingTypesSize = 0;

  size_t TemplateTypeParmTypesSize = 0;
  size_t ObjCTypeParamTypesSize = 0;
  size_t SubstTemplateTypeParmTypesSize = 0;
  size_t SubstTemplateTypeParmPackTypesSize = 0;
  size_t SubstBuiltinTemplatePackTypesSize = 0;

  size_t TemplateSpecializationTypesSize = 0;
  size_t ParenTypesSize = 0;
  size_t TagTypesSize = 0;
  size_t UnresolvedUsingTypesSize = 0;
  size_t UsingTypesSize = 0;
  size_t TypedefTypesSize = 0;
  size_t DependentNameTypesSize = 0;
  size_t PackExpansionTypesSize = 0;
  size_t UnaryTransformTypesSize = 0;

  size_t AutoTypesSize = 0;
  size_t DeducedTemplateSpecializationTypesSize = 0;
  size_t AtomicTypesSize = 0;
  size_t AttributedTypesSize = 0;
  size_t PipeTypesSize = 0;
  size_t BitIntTypesSize = 0;
  size_t DependentBitIntTypesSize = 0;
  size_t BTFTagAttributedTypesSize = 0;
  size_t HLSLAttributedResourceTypesSize = 0;
  size_t HLSLInlineSpirvTypesSize = 0;

  size_t CountAttributedTypesSize = 0;

  size_t QualifiedTemplateNamesSize = 0;
  size_t DependentTemplateNamesSize = 0;
  size_t SubstTemplateTemplateParmsSize = 0;
  size_t SubstTemplateTemplateParmPacksSize = 0;
  size_t DeducedTemplatesSize = 0;

  size_t ArrayParameterTypesSize = 0;

  size_t PredefinedSugarTypesSize = 0;

  size_t NamespaceAndPrefixStoragesSize = 0;

  size_t ASTRecordLayoutsSize = 0;

  size_t MemoizedTypeInfoSize = 0;

  size_t MemoizedUnadjustedAlignSize = 0;

  size_t KeyFunctionsSize = 0;

  size_t BlockVarCopyInitsSize = 0;

  size_t MSGuidDeclsSize = 0;

  size_t UnnamedGlobalConstantDeclsSize = 0;

  size_t TemplateParamObjectDeclsSize = 0;

  size_t StringLiteralCacheSize = 0;

  size_t DestroyingOperatorDeletesSize = 0;
  size_t TypeAwareOperatorNewAndDeletesSize = 0;

  size_t OperatorDeletesForVirtualDtorSize = 0;

  size_t GlobalOperatorDeletesForVirtualDtorSize = 0;

  size_t ArrayOperatorDeletesForVirtualDtorSize = 0;
  size_t GlobalArrayOperatorDeletesForVirtualDtorSize = 0;

  size_t MaybeRequireVectorDeletingDtorSize = 0;

  size_t MergedDeclsSize = 0;
  size_t DeclAttrsSize = 0;
  size_t MergedDefModulesSize = 0;

  size_t ModuleInitializersSize = 0;
  size_t PrimaryModuleNameMapSize = 0;
  size_t SameModuleLookupSetSize = 0;

  size_t ScalableVecTyMapSize = 0;
  size_t LambdaCastPathsSize = 0;
  size_t RawCommentsSize = 0;
  size_t RedeclChainCommentsSize = 0;
  size_t CommentlessRedeclChainsSize = 0;
  size_t ParsedCommentsSize = 0;
  size_t RelocatableClassesSize = 0;
  size_t ParamIndicesSize = 0;
  size_t MangleNumbersSize = 0;
  size_t StaticLocalNumbersSize = 0;
  size_t TemplateOrInstantiationSize = 0;

  size_t InstantiatedFromUsingDeclSize = 0;
  size_t InstantiatedFromUsingEnumDeclSize = 0;

  size_t InstantiatedFromUsingShadowDeclSize = 0;

  size_t InstantiatedFromUnnamedFieldDeclSize = 0;
  size_t OverriddenMethodsSize = 0;
  size_t MangleNumberingContextsSize = 0;
  size_t ExtraMangleNumberingContextsSize = 0;

  size_t TraversalScopeSize = 0;

  /// object-c
  size_t ObjCObjectTypesSize = 0;
  size_t ObjCObjectPointerTypesSize = 0;

  mutable TypedefDecl *ObjCIdDeclCP = nullptr;

  mutable TypedefDecl *ObjCSelDeclCP = nullptr;

  mutable TypedefDecl *ObjCClassDeclCP = nullptr;

  mutable ObjCInterfaceDecl *ObjCProtocolClassDeclCP = nullptr;

  // llvm::PointerIntPair<StoredDeclsMap *, 1> LastSDM;
  //    = llvm::PointerIntPair<StoredDeclsMap *, 1>(nullptr, 0);
  mutable QualType AutoDeductTy;     // Deduction against 'auto'.
  mutable QualType AutoRRefDeductTy; // Deduction against 'auto &&'.

  // mutable DeclarationNameTable DeclarationNames; need to revert.
  struct DeclarationNamesCP {
    size_t CXXConstructorNamesSize = 0;
    size_t CXXDestructorNamesSize = 0;
    size_t CXXConversionFunctionNamesSize = 0;
    detail::CXXOperatorIdName CXXOperatorNamesCP[NUM_OVERLOADED_OPERATORS];

    size_t CXXLiteralOperatorNamesSize = 0;

    size_t CXXDeductionGuideNamesSize = 0;
  };

  DeclarationNamesCP DeclarationNames;
};

class ASTContextStateRecovery {
private:
  ASTContext &Ctx;

public:
  explicit ASTContextStateRecovery(ASTContext &Ctx) : Ctx(Ctx) {}

  ASTContextStateRecovery(const ASTContextStateRecovery &) = delete;
  ASTContextStateRecovery &operator=(const ASTContextStateRecovery &) = delete;

  void stash(StashCheckPoint &CP);
  void restore(StashCheckPoint &CP, llvm::SlabCheckPoint SlabCP);
  void commit();
};

/// Index of a per PTU State.
using PTUID = unsigned;

// A snapshot of the handful of DefinitionData bits/bitfields that can
// change after a class (or class template specialization -- the two are
// handled identically here since a specialization is non-dependent by the
// time it's instantiated) is otherwise "done" being defined.
//
// These fields only move when the compiler generates implicit special
// members.
//
// This footprint exists purely to undo the effects of implicit special
// member generation.
struct DefinitionDataFootprint {
  unsigned Aggregate : 1;
  unsigned PlainOldData : 1;
  unsigned Empty : 1;
  unsigned Polymorphic : 1;
  unsigned IsStandardLayout : 1;
  unsigned IsCXX11StandardLayout : 1;
  unsigned HasTrivialSpecialMembers : 6;
  unsigned HasTrivialSpecialMembersForCall : 6;
  unsigned DeclaredNonTrivialSpecialMembers : 6;
  unsigned DeclaredNonTrivialSpecialMembersForCall : 6;
  unsigned HasIrrelevantDestructor : 1;
  unsigned HasConstexprNonCopyMoveConstructor : 1;
  unsigned HasDefaultedDefaultConstructor : 1;
  unsigned HasConstexprDefaultConstructor : 1;
  unsigned HasDeclaredCopyConstructorWithConstParam : 1;
  unsigned HasDeclaredCopyAssignmentWithConstParam : 1;
  unsigned IsAnyDestructorNoReturn : 1;
  unsigned DeclaredSpecialMembers : 6;

  bool operator==(const DefinitionDataFootprint &O) const {
    return Aggregate == O.Aggregate && PlainOldData == O.PlainOldData &&
           Empty == O.Empty && Polymorphic == O.Polymorphic &&
           IsStandardLayout == O.IsStandardLayout &&
           IsCXX11StandardLayout == O.IsCXX11StandardLayout &&
           HasTrivialSpecialMembers == O.HasTrivialSpecialMembers &&
           HasTrivialSpecialMembersForCall ==
               O.HasTrivialSpecialMembersForCall &&
           DeclaredNonTrivialSpecialMembers ==
               O.DeclaredNonTrivialSpecialMembers &&
           DeclaredNonTrivialSpecialMembersForCall ==
               O.DeclaredNonTrivialSpecialMembersForCall &&
           HasIrrelevantDestructor == O.HasIrrelevantDestructor &&
           HasConstexprNonCopyMoveConstructor ==
               O.HasConstexprNonCopyMoveConstructor &&
           HasDefaultedDefaultConstructor == O.HasDefaultedDefaultConstructor &&
           HasConstexprDefaultConstructor == O.HasConstexprDefaultConstructor &&
           HasDeclaredCopyConstructorWithConstParam ==
               O.HasDeclaredCopyConstructorWithConstParam &&
           HasDeclaredCopyAssignmentWithConstParam ==
               O.HasDeclaredCopyAssignmentWithConstParam &&
           IsAnyDestructorNoReturn == O.IsAnyDestructorNoReturn &&
           DeclaredSpecialMembers == O.DeclaredSpecialMembers;
  }
  bool operator!=(const DefinitionDataFootprint &O) const {
    return !(*this == O);
  }

  // void update(const CXXRecordDecl::DefinitionData &Live) {
  //   Aggregate = Live.Aggregate;
  //   PlainOldData = Live.PlainOldData;
  //   Empty = Live.Empty;
  //   Polymorphic = Live.Polymorphic;
  //   IsStandardLayout = Live.IsStandardLayout;
  //   IsCXX11StandardLayout = Live.IsCXX11StandardLayout;
  //   HasTrivialSpecialMembers = Live.HasTrivialSpecialMembers;
  //   HasTrivialSpecialMembersForCall = Live.HasTrivialSpecialMembersForCall;
  //   DeclaredNonTrivialSpecialMembers = Live.DeclaredNonTrivialSpecialMembers;
  //   DeclaredNonTrivialSpecialMembersForCall =
  //       Live.DeclaredNonTrivialSpecialMembersForCall;
  //   HasIrrelevantDestructor = Live.HasIrrelevantDestructor;
  //   HasConstexprNonCopyMoveConstructor =
  //       Live.HasConstexprNonCopyMoveConstructor;
  //   HasDefaultedDefaultConstructor = Live.HasDefaultedDefaultConstructor;
  //   HasConstexprDefaultConstructor = Live.HasConstexprDefaultConstructor;
  //   HasDeclaredCopyConstructorWithConstParam =
  //       Live.HasDeclaredCopyConstructorWithConstParam;
  //   HasDeclaredCopyAssignmentWithConstParam =
  //       Live.HasDeclaredCopyAssignmentWithConstParam;
  //   IsAnyDestructorNoReturn = Live.IsAnyDestructorNoReturn;
  //   DeclaredSpecialMembers = Live.DeclaredSpecialMembers;
  // }

  // void restore(CXXRecordDecl::DefinitionData &Live) const {
  //   Live.Aggregate = Aggregate;
  //   Live.PlainOldData = PlainOldData;
  //   Live.Empty = Empty;
  //   Live.Polymorphic = Polymorphic;
  //   Live.IsStandardLayout = IsStandardLayout;
  //   Live.IsCXX11StandardLayout = IsCXX11StandardLayout;
  //   Live.HasTrivialSpecialMembers = HasTrivialSpecialMembers;
  //   Live.HasTrivialSpecialMembersForCall = HasTrivialSpecialMembersForCall;
  //   Live.DeclaredNonTrivialSpecialMembers = DeclaredNonTrivialSpecialMembers;
  //   Live.DeclaredNonTrivialSpecialMembersForCall =
  //       DeclaredNonTrivialSpecialMembersForCall;
  //   Live.HasIrrelevantDestructor = HasIrrelevantDestructor;
  //   Live.HasConstexprNonCopyMoveConstructor =
  //       HasConstexprNonCopyMoveConstructor;
  //   Live.HasDefaultedDefaultConstructor = HasDefaultedDefaultConstructor;
  //   Live.HasConstexprDefaultConstructor = HasConstexprDefaultConstructor;
  //   Live.HasDeclaredCopyConstructorWithConstParam =
  //       HasDeclaredCopyConstructorWithConstParam;
  //   Live.HasDeclaredCopyAssignmentWithConstParam =
  //       HasDeclaredCopyAssignmentWithConstParam;
  //   Live.IsAnyDestructorNoReturn = IsAnyDestructorNoReturn;
  //   Live.DeclaredSpecialMembers = DeclaredSpecialMembers;
  // }
};

// Stores the state of a class template specialization. Tracks its
// specialization kind, point of instantiation, source location, and lexical
// declaration context so the state can be compared and restored.
struct SpecializationFootprint {
  unsigned SpecializationKind : 3;
  SourceLocation PointOfInstantiation;
  SourceLocation Location;
  const DeclContext *LexicalDC;

  bool operator==(const SpecializationFootprint &O) const {
    return SpecializationKind == O.SpecializationKind &&
           PointOfInstantiation == O.PointOfInstantiation &&
           Location == O.Location && LexicalDC == O.LexicalDC;
  }
  bool operator!=(const SpecializationFootprint &O) const {
    return !(*this == O);
  }

  void update(const ClassTemplateSpecializationDecl &Live) {
    SpecializationKind = Live.getSpecializationKind();
    PointOfInstantiation = Live.getPointOfInstantiation();
    Location = Live.getLocation();
    LexicalDC = Live.getLexicalDeclContext();
  }

  void restore(ClassTemplateSpecializationDecl &Live) const {
    Live.setSpecializationKind(
        static_cast<TemplateSpecializationKind>(SpecializationKind));
    Live.setPointOfInstantiation(PointOfInstantiation);
    Live.setLocation(Location);
    Live.setLexicalDeclContext(const_cast<DeclContext *>(LexicalDC));
  }
};

// Stores the state of a variable template specialization. Tracks its
// specialization kind and point of instantiation so the state can be
// compared and restored.
struct VarSpecializationFootprint {
  unsigned SpecializationKind : 3;
  SourceLocation PointOfInstantiation;

  bool operator==(const VarSpecializationFootprint &O) const {
    return SpecializationKind == O.SpecializationKind &&
           PointOfInstantiation == O.PointOfInstantiation;
  }
  bool operator!=(const VarSpecializationFootprint &O) const {
    return !(*this == O);
  }

  void update(const VarTemplateSpecializationDecl &Live) {
    SpecializationKind = Live.getSpecializationKind();
    PointOfInstantiation = Live.getPointOfInstantiation();
  }

  void restore(VarTemplateSpecializationDecl &Live) const {
    Live.setSpecializationKind(
        static_cast<TemplateSpecializationKind>(SpecializationKind));
    Live.setPointOfInstantiation(PointOfInstantiation);
  }
};

// Stores the state of a function template specialization. Tracks its
// specialization kind, point of instantiation, source location, lexical
// declaration context, and constexpr state so the state can be compared
// and restored.
struct FunctionSpecializationFootprint {
  unsigned SpecializationKind : 3;
  SourceLocation PointOfInstantiation;
  SourceLocation Location;
  const DeclContext *LexicalDC;
  ConstexprSpecKind ConstexprKind;

  bool operator==(const FunctionSpecializationFootprint &O) const {
    return SpecializationKind == O.SpecializationKind &&
           PointOfInstantiation == O.PointOfInstantiation &&
           Location == O.Location && LexicalDC == O.LexicalDC &&
           ConstexprKind == O.ConstexprKind;
  }
  bool operator!=(const FunctionSpecializationFootprint &O) const {
    return !(*this == O);
  }

  void update(const FunctionDecl &Live) {
    SpecializationKind = Live.getTemplateSpecializationKind();
    PointOfInstantiation = Live.getPointOfInstantiation();
    Location = Live.getLocation();
    LexicalDC = Live.getLexicalDeclContext();
    ConstexprKind = Live.getConstexprKind();
  }

  void restore(FunctionDecl &Live) const {
    Live.setTemplateSpecializationKind(
        static_cast<TemplateSpecializationKind>(SpecializationKind),
        PointOfInstantiation);
    Live.setLocation(Location);
    Live.setLexicalDeclContext(const_cast<DeclContext *>(LexicalDC));
    Live.setConstexprKind(ConstexprKind);
  }
};

// Stores the state of an ordinary class member created from a class template
// instantiation. Tracks its specialization kind, point of instantiation, and
// source location so the state can be compared and restored.
struct MemberSpecializationFootprint {
  unsigned SpecializationKind : 3;
  SourceLocation PointOfInstantiation;
  SourceLocation Location;

  bool operator==(const MemberSpecializationFootprint &O) const {
    return SpecializationKind == O.SpecializationKind &&
           PointOfInstantiation == O.PointOfInstantiation &&
           Location == O.Location;
  }
  bool operator!=(const MemberSpecializationFootprint &O) const {
    return !(*this == O);
  }

  template <typename OwnerT> void update(const OwnerT &Live) {
    const MemberSpecializationInfo *MSI = Live.getMemberSpecializationInfo();
    SpecializationKind = MSI->getTemplateSpecializationKind();
    PointOfInstantiation = MSI->getPointOfInstantiation();
    Location = Live.getLocation();
  }

  template <typename OwnerT> void restore(OwnerT &Live) const {
    MemberSpecializationInfo *MSI = Live.getMemberSpecializationInfo();
    MSI->setTemplateSpecializationKind(
        static_cast<TemplateSpecializationKind>(SpecializationKind));
    MSI->setPointOfInstantiation(PointOfInstantiation);
    Live.setLocation(Location);
  }
};

class ASTStateReader {
public:
  static DefinitionDataFootprint *
  createDefinitionDataFootprint(const ASTContext &Ctx, const CXXRecordDecl &RD);

  static bool compareDefinitionDataFootprint(const DefinitionDataFootprint &FP,
                                             const CXXRecordDecl &RD);

  static void restoreDefinitionDataFootprint(const DefinitionDataFootprint &FP,
                                             CXXRecordDecl &RD);

  static SpecializationFootprint *
  createSpecializationFootprint(const ASTContext &Ctx,
                                const ClassTemplateSpecializationDecl &Spec);

  static bool
  compareSpecializationFootprint(const SpecializationFootprint &FP,
                                 const ClassTemplateSpecializationDecl &Spec) {
    SpecializationFootprint Live;
    Live.update(Spec);
    return FP == Live;
  }
  static void
  restoreSpecializationFootprint(const SpecializationFootprint &FP,
                                 ClassTemplateSpecializationDecl &Spec) {
    FP.restore(Spec);
  }

  static VarSpecializationFootprint *
  createVarSpecializationFootprint(const ASTContext &Ctx,
                                   const VarTemplateSpecializationDecl &Spec);

  static bool
  compareVarSpecializationFootprint(const VarSpecializationFootprint &FP,
                                    const VarTemplateSpecializationDecl &Spec) {
    VarSpecializationFootprint Live;
    Live.update(Spec);
    return FP == Live;
  }
  static void
  restoreVarSpecializationFootprint(const VarSpecializationFootprint &FP,
                                    VarTemplateSpecializationDecl &Spec) {
    FP.restore(Spec);
  }

  static FunctionSpecializationFootprint *
  createFunctionSpecializationFootprint(const ASTContext &Ctx,
                                        const FunctionDecl &FD);

  static bool compareFunctionSpecializationFootprint(
      const FunctionSpecializationFootprint &FP, const FunctionDecl &FD);

  static void restoreFunctionSpecializationFootprint(
      const FunctionSpecializationFootprint &FP, FunctionDecl &FD) {
    FP.restore(FD);
  }

  template <typename OwnerT>
  static MemberSpecializationFootprint *
  createMemberSpecializationFootprint(const ASTContext &Ctx, const OwnerT &D);

  template <typename OwnerT>
  static bool
  compareMemberSpecializationFootprint(const MemberSpecializationFootprint &FP,
                                       const OwnerT &D);
  template <typename OwnerT>
  static void
  restoreMemberSpecializationFootprint(const MemberSpecializationFootprint &FP,
                                       OwnerT &D);

  static void restoreDefinitionAndRevertDC(CXXRecordDecl &RD);

  static void revertDefinitionArrival(Decl &D);

  static const void *getRawCommonPtr(const RedeclarableTemplateDecl &RT);

  static bool isTemplateCanonInjectedTSTValid(const ClassTemplateDecl *CTD);

  static void resetTemplateCommonBase(RedeclarableTemplateDecl &RT);

  static void resetCanonInjectedTST(ClassTemplateDecl &CTD);

  static const Type *getRawTypeForDecl(const TypeDecl *TD);

  static void resetTypeForDecl(TypeDecl *TD);
};

template <typename DataT> struct Snapshot {
  PTUID ID;
  DataT *Data;
};

template <typename DataT> class StateAwareChain {
  llvm::SmallVector<Snapshot<DataT>, 4> History;

public:
  bool empty() const { return History.empty(); }

  const DataT *mostRecent() const {
    return History.empty() ? nullptr : History.back().Data;
  }

  std::optional<PTUID> mostRecentID() const {
    return History.empty() ? std::nullopt
                           : std::optional<PTUID>(History.back().ID);
  }

  std::optional<PTUID> oldestID() const {
    return History.empty() ? std::nullopt
                           : std::optional<PTUID>(History.front().ID);
  }

  void commit(PTUID ID, DataT *Fresh) {
    if (!History.empty() && *History.back().Data == *Fresh)
      return;
    assert(History.back().ID != ID);
    History.push_back(Snapshot<DataT>{ID, Fresh});
  }

  const DataT *getPrevious(PTUID ID) const {
    for (auto It = History.rbegin(); It != History.rend(); ++It)
      if (It->ID < ID)
        return It->Data;
    return nullptr;
  }

  /// Pure removal: drop every entry with ID >= \p ID (LIFO). Never
  /// restores anything -- callers must read mostRecent()/getPrevious()
  /// themselves first if they need the value about to be dropped.
  void removeFrom(PTUID ID) {
    while (!History.empty() && History.back().ID >= ID)
      History.pop_back(); // no delete.
  }
};

// struct UnusedTrait {};

using RecordDeclDefinitionDataChain = StateAwareChain<DefinitionDataFootprint>;
using SpecializationChain = StateAwareChain<SpecializationFootprint>;
using VarSpecializationChain = StateAwareChain<VarSpecializationFootprint>;
using FunctionSpecializationChain =
    StateAwareChain<FunctionSpecializationFootprint>;
// One chain type shared by all four MemberSpecializationInfo-backed owner
// kinds (Function/Var/Record/Enum) -- the footprint shape is identical
// across all four (see MemberSpecializationFootprint's own comment), so
// unlike the three chains above, a single alias is enough; what varies is
// only the DenseMap key type in IncrementalStateTracker
// (FunctionDecl*/VarDecl*/ CXXRecordDecl*/EnumDecl*), one map per owner kind.
using MemberSpecializationChain =
    StateAwareChain<MemberSpecializationFootprint>;

class PTUCheckpointLedger {
  llvm::SmallVector<llvm::SlabCheckPoint, 16> CheckpointBeforePTU;

public:
  /// Called once, right before parsing PTU \p ID begins.
  void recordCheckpoint(PTUID ID, llvm::SlabCheckPoint CP) {
    assert(ID == CheckpointBeforePTU.size() &&
           "PTUs must be recorded in order");
    CheckpointBeforePTU.push_back(CP);
  }

  /// \return the PTU that allocated \p Ptr, or std::nullopt if \p Ptr
  /// predates the oldest recorded checkpoint.
  ///
  /// Walks newest-to-oldest: checkpoints only ever move forward for state
  /// that has survived (committed PTUs are never rewound), so the first
  /// checkpoint for which \p Ptr is "after" is the PTU that produced it.
  std::optional<PTUID> attribute(const ASTContext &Ctx, const void *Ptr) const {
    for (PTUID ID = CheckpointBeforePTU.size(); ID-- > 0;) {
      if (Ctx.getAllocator().isAfterCheckpoint(Ptr, CheckpointBeforePTU[ID]))
        return ID;
    }
    return std::nullopt;
  }

  inline std::optional<PTUID> attributeByAddress(const ASTContext &Ctx,
                                                 const void *Ptr) {
    // Binary search for the largest ID whose checkpoint Ptr is after -- i.e.
    // the newest PTU boundary this address was allocated on or past.
    size_t Lo = 0, Hi = CheckpointBeforePTU.size();
    std::optional<PTUID> Result;
    while (Lo < Hi) {
      size_t Mid = Lo + (Hi - Lo) / 2;
      if (Ctx.getAllocator().isAfterCheckpoint(Ptr, CheckpointBeforePTU[Mid])) {
        Result = static_cast<PTUID>(Mid);
        Lo = Mid + 1; // still after Mid's checkpoint -- look for a later one
      } else {
        Hi = Mid; // not even after Mid -- must be before it
      }
    }
    return Result;
  }

  bool predatesPTU(const ASTContext &Ctx, const void *Ptr, PTUID ID) const {
    assert(ID < CheckpointBeforePTU.size());
    return !Ctx.getAllocator().isAfterCheckpoint(Ptr, CheckpointBeforePTU[ID]);
  }

  bool isFromThisPTU(const ASTContext &Ctx, const void *Ptr, PTUID ID) const {
    return Ctx.getAllocator().isAfterCheckpoint(Ptr, CheckpointBeforePTU[ID]);
  }

  /// Drop checkpoints from \p ID onward.
  void undoFrom(PTUID ID) {
    if (ID < CheckpointBeforePTU.size())
      CheckpointBeforePTU.resize(ID);
  }
};

/// need to handle this     TagDeclBitfields TagDeclBits;

template <typename ValueT> struct FieldMutation {
  PTUID ID;
  ValueT OldValue;
};

template <typename OwnerT, typename ValueT> class FieldMutationChain {
  llvm::DenseMap<const OwnerT *, llvm::SmallVector<FieldMutation<ValueT>, 2>>
      Log;

public:
  void noteMutation(PTUID ID, const OwnerT *Owner, ValueT OldValue) {
    auto &Entries = Log[Owner];
    if (!Entries.empty() && Entries.back().ID == ID)
      return;
    Entries.push_back(FieldMutation<ValueT>{ID, OldValue});
  }

  const ValueT *mostRecent(const OwnerT *Owner) const {
    auto It = Log.find(Owner);
    return (It == Log.end() || It->second.empty())
               ? nullptr
               : &It->second.back().OldValue;
  }

  std::optional<PTUID> mostRecentID(const OwnerT *Owner) const {
    auto It = Log.find(Owner);
    if (It == Log.end())
      return std::nullopt;
    auto &History = It->second;
    return History.empty() ? std::nullopt
                           : std::optional<PTUID>(History.back().ID);
  }

  std::optional<PTUID> oldestID(const OwnerT *Owner) const {
    auto It = Log.find(Owner);
    if (It == Log.end())
      return std::nullopt;
    auto &History = It->second;
    return History.empty() ? std::nullopt
                           : std::optional<PTUID>(History.front().ID);
  }

  template <typename FnT> void forEachOwnerSince(PTUID ID, FnT &&Fn) const {
    for (auto &Entry : Log)
      if (!Entry.second.empty() && Entry.second.back().ID >= ID)
        Fn(Entry.first);
  }

  void removeFrom(PTUID ID) {
    llvm::SmallVector<OwnerT *> ToErase;
    for (auto &Entry : Log) {
      auto &Entries = Entry.second;
      while (!Entries.empty() && Entries.back().ID >= ID)
        Entries.pop_back();
      if (Entries.empty())
        ToErase.push_back(Entry.first);
    }
    for (const OwnerT *O : ToErase)
      Log.erase(O);
  }

  void forget(const OwnerT *Owner) { Log.erase(Owner); }
};

// struct FunctionTypeEntry {
//   QualType Type;
//   QualType OverridingType;
// };
using FunctionExceptionSpecChain = FieldMutationChain<FunctionDecl, QualType>;
using TypeForDeclChain = FieldMutationChain<TagDecl, const Type *>;
// using CanonInjectedTSTChain =
//     FieldMutationChain<ClassTemplateDecl, CanQualType>;

struct MutationRecord {
  enum class DeclShape : uint16_t {
    None = 0,
    // Base shapes -- what the decl fundamentally IS. Exactly one is set.
    Class = 1 << 0,    // CXXRecordDecl/TagDecl: DefinitionData, TypeForDecl
    Function = 1 << 1, // FunctionDecl: exception spec, deduced return, body
    Var = 1 << 2,      // VarDecl: cached constant-eval result
    Enum = 1 << 3,     // EnumDecl: TypeForDecl
    Template = 1 << 4, // RedeclarableTemplateDecl: spec list grows
    Typedef = 1 << 5,  // TypedefDecl/TypeAliasDecl: TypeForDecl.
  };

  enum class MutationKind : uint32_t {
    // ---- Common ----
    DefinitionInstantiate = 1 << 0, // Class, Function, Var
    SpecInfo = 1 << 1,              // Class, Function, Var  (Spec)
    MemberSpecInfo = 1 << 2,        // Class, Function, Var, Enum (Member)
    TypeForDecl = 1 << 3,           // Class, Enum

    // ---- Shape-specific ----
    DefinitionData = 1 << 8,       // Class only
    ExceptionSpec = 1 << 9,        // Function only
    DeducedReturnType = 1 << 10,   // Function only
    EvaluatedValue = 1 << 11,      // Var only
    SpecializationAdded = 1 << 12, // Template only

    TemplateCommon = 1 << 4,   // Template CommonBase
    CanonInjectedTST = 1 << 5, // Template CommonBase Type
    None = 1 << 24,
  };

  DeclShape S = DeclShape::None;
  uint32_t MutationType = 0; // what this decl can ever have

  void add(MutationKind K) { MutationType |= uint32_t(K); }
  void add(uint32_t K) { MutationType |= uint32_t(K); }
  bool has(MutationKind K) const { return MutationType & uint32_t(K); }
  void clear(MutationKind K) { MutationType &= ~uint32_t(K); }
};

using DeclShape = MutationRecord::DeclShape;
using MutationType = MutationRecord::MutationKind;

class PTUMutationActions;

struct PTUStateInfo {
  PTUID ID;
  const TranslationUnitDecl *ThisTU; // current info

  llvm::MapVector<const Decl *, MutationRecord> Mutations;

  // struct CreationRecord {
  //   DeclShape S;
  //   CreatedType Type;
  // };

  // llvm::MapVector<const Decl *, CreationRecord> CreatedDecls;
  llvm::SmallPtrSet<const Decl *, 4> ImplicitDecls;

  /// here touched info mean other this belongs to other PTUs;
  llvm::SmallPtrSet<const DeclContext *, 4> TouchedDC;

  template <typename KindT>
  void noteMutated(const Decl *D, DeclShape S, KindT K) {
    if (!D->isDefinedOutsideFunctionOrMethod())
      return;
    auto [It, Inserted] = Mutations.try_emplace(D);
    if (Inserted) {
      assert(It->second.S == S && "same decl noted under two different shapes");
      It->second.S = S;
    }
    It->second.add(K);
  }

  void verifyMutations();

  // False until this PTU is actually committed.
  // PTUMutationActions uses this to decide what "undo" should do:
  // - If this PTU was never committed, nothing was added to the chain,
  //   so we just restore the last PTU that was committed.
  // - If this PTU was already committed (for example, by using %undo),
  //   first remove the entries created by this PTU, then restore the
  //   state from before this PTU was added.
  bool Commited = false;
};

//===----------------------------------------------------------------------===//
// SweepTracker -- Tracks declarations whose mutations cannot be reported
// directly by an ASTMutationListener.
//
// Each Decl is tracked with the hidden mutation kinds that are still possible.
// At commit time, sweep() checks the tracked kinds to find mutations made by
// the current PTU. Once a mutation kind is confirmed or can no longer happen,
// it is removed from tracking.
//
// During restore, a mutation kind can be tracked again so its mutation sites
// can be restored when needed.
//===----------------------------------------------------------------------===//
class SweepTracker {
  llvm::DenseMap<const Decl *, uint32_t> Active;

public:
  void track(const Decl *D, uint32_t HiddenBits) {
    if (HiddenBits)
      Active[D] |= HiddenBits;
  }

  bool isTrackedFor(const Decl *D, uint32_t Flag) const {
    auto It = Active.find(D);
    return It != Active.end() && (It->second & Flag);
  }

  void settle(const Decl *D, uint32_t Flag) {
    auto It = Active.find(D);
    if (It == Active.end())
      return;
    It->second &= ~Flag;
    if (!It->second)
      Active.erase(It);
  }

  // Call at commit(ID) time. OnConfirmed is invoked as
  // (const Decl *D, DeclShape S, uint32_t ConfirmedKinds) for every decl
  // that had something newly confirmed this sweep.
  template <typename OnConfirmedFn>
  void sweep(PTUID ID, OnConfirmedFn &&OnConfirmed);
};

//===----------------------------------------------------------------------===//
// DeclLinkedState / LinkedDeclNodeGenerator
//
// These nodes keep the footprint-chain state associated with a Decl.
// Each Decl kind has its own node type so the node only contains state that
// is valid for that kind. DeclLinkedState then uses the node type to identify
// which kind of Decl it belongs to.
//
// The generator owns these nodes and their chains and reuses them through
// per-type free lists. This is intentional: nodes need to be released and
// reused, so they use normal heap allocation instead of ASTContext's
// bump allocator.
//
// The goal is to keep the linked state small, type-safe, and reusable without
// adding fields for states that a particular Decl can never have.
//===----------------------------------------------------------------------===//
struct CXXClassDeclNode {
  PTUID OriginID;
  RecordDeclDefinitionDataChain *DefData = nullptr;
  llvm::PointerUnion<SpecializationChain *, MemberSpecializationChain *> Spec;
  CXXClassDeclNode *Next = nullptr; // free-list link, meaningless off the list
};

struct VarDeclNode {
  PTUID OriginID;
  llvm::PointerUnion<VarSpecializationChain *, MemberSpecializationChain *>
      Spec;
  VarDeclNode *Next = nullptr;
};

struct FunctionDeclNode {
  PTUID OriginID;
  llvm::PointerUnion<FunctionSpecializationChain *, MemberSpecializationChain *>
      Spec;
  FunctionDeclNode *Next = nullptr;
};

struct EnumDeclNode {
  PTUID OriginID;
  MemberSpecializationChain *MemberSpec =
      nullptr; // the only thing an enum can ever have
  EnumDeclNode *Next = nullptr;
};

using DeclLinkedState = llvm::PointerUnion<CXXClassDeclNode *, VarDeclNode *,
                                           FunctionDeclNode *, EnumDeclNode *>;

/// Intrusive free-list pool for the four node types above. Released nodes
/// are kept for reuse instead of being immediately deleted. The pool is
/// bounded by Capacity; once it is full, additional released nodes are
/// deleted instead of being kept. This keeps memory usage bounded while
/// still allowing freed nodes to be reused.

template <typename T> class NodePool {
  T *FreeList = nullptr;
  unsigned FreeCount = 0;
  unsigned Capacity;

public:
  explicit NodePool(unsigned Capacity = 64) : Capacity(Capacity) {}

  ~NodePool() {
    while (FreeList) {
      T *Dead = FreeList;
      FreeList = FreeList->Next;
      delete Dead;
    }
  }
  /// Hands back a reset (all-default) node -- recycled if one is free,
  /// freshly allocated otherwise.
  T *acquire() {
    if (T *N = FreeList) {
      FreeList = N->Next;
      --FreeCount;
      *N = T();
      return N;
    }
    return new T();
  }
  /// Takes ownership back. Kept for a future acquire() to hand out again
  /// if the free list is under capacity; reclaimed for real (delete)
  /// otherwise, so this pool never holds more than Capacity dead nodes.
  void release(T *N) {
    if (FreeCount >= Capacity) {
      delete N;
      return;
    }
    N->Next = FreeList;
    FreeList = N;
    ++FreeCount;
  }
};

/// Pool for reusing chain objects without modifying StateAwareChain.
/// Released chains are kept for reuse, and the pool is bounded by Capacity
/// so unused chains do not cause unbounded memory growth.
template <typename ChainT> class ChainPool {
  llvm::SmallVector<ChainT *, 8> Free;
  unsigned Capacity;

public:
  explicit ChainPool(unsigned Capacity = 64) : Capacity(Capacity) {}

  ~ChainPool() {
    for (ChainT *C : Free)
      delete C;
  }

  ChainT *acquire() {
    if (!Free.empty()) {
      ChainT *C = Free.pop_back_val();
      *C = ChainT();
      return C;
    }
    return new ChainT();
  }
  /// Same capacity rule as NodePool::release() -- reclaimed for real once
  /// Free is at capacity, rather than growing without bound.
  void release(ChainT *C) {
    if (Free.size() >= Capacity) {
      delete C;
      return;
    }
    Free.push_back(C);
  }
};

class LinkedDeclNodeGenerator {
  NodePool<CXXClassDeclNode> RecordNodes;
  NodePool<VarDeclNode> VarNodes;
  NodePool<FunctionDeclNode> FunctionNodes;
  NodePool<EnumDeclNode> EnumNodes;

  ChainPool<RecordDeclDefinitionDataChain> DefDataChains;
  ChainPool<SpecializationChain> ClassSpecChains;
  ChainPool<VarSpecializationChain> VarSpecChains;
  ChainPool<FunctionSpecializationChain> FunctionSpecChains;
  ChainPool<MemberSpecializationChain> MemberSpecChains;

public:
  CXXClassDeclNode *acquireRecordNode() { return RecordNodes.acquire(); }
  VarDeclNode *acquireVarNode() { return VarNodes.acquire(); }
  FunctionDeclNode *acquireFunctionNode() { return FunctionNodes.acquire(); }
  EnumDeclNode *acquireEnumNode() { return EnumNodes.acquire(); }

  RecordDeclDefinitionDataChain *acquireDefDataChain() {
    return DefDataChains.acquire();
  }
  SpecializationChain *acquireClassSpecChain() {
    return ClassSpecChains.acquire();
  }
  VarSpecializationChain *acquireVarSpecChain() {
    return VarSpecChains.acquire();
  }
  FunctionSpecializationChain *acquireFunctionSpecChain() {
    return FunctionSpecChains.acquire();
  }
  MemberSpecializationChain *acquireMemberSpecChain() {
    return MemberSpecChains.acquire();
  }

  // One release() name per type, dispatched by overload resolution --
  // the caller doesn't need to know which internal pool a given pointer
  // belongs to.
  void release(CXXClassDeclNode *N) { RecordNodes.release(N); }
  void release(VarDeclNode *N) { VarNodes.release(N); }
  void release(FunctionDeclNode *N) { FunctionNodes.release(N); }
  void release(EnumDeclNode *N) { EnumNodes.release(N); }
  void release(RecordDeclDefinitionDataChain *C) { DefDataChains.release(C); }
  void release(SpecializationChain *C) { ClassSpecChains.release(C); }
  void release(VarSpecializationChain *C) { VarSpecChains.release(C); }
  void release(FunctionSpecializationChain *C) {
    FunctionSpecChains.release(C);
  }
  void release(MemberSpecializationChain *C) { MemberSpecChains.release(C); }
};

enum class LangMode : uint8_t {
  C,   // no DefinitionData, no templates, no implicit special members
  CXX, // full set
};

// Overall flow:
// - Track changes made by other PTUs so they can be applied to the relevant
//   chains/mutations when those PTUs are committed.
// - On rollback of a committed PTU, remove the entries created by that PTU
//   and restore the state that existed before it.
// - On rollback of a PTU that was never committed, restore the chain to its
//   current mostRecent() state, since this PTU never added anything to it.
// - For the current PTU, collect newly added declarations that are
//   modifiable/visible to other PTUs. Local-only declarations are not exposed.
//
// In short, this keeps the shared declaration state in sync across PTUs while
// keeping declarations that are local to the current PTU private.
class IncrementalStateTracker {
private:
  ASTContext &Ctx;
  LangMode Mode;
  PTUID NextID = 0;
  PTUID CurID = NextID;
  PTUCheckpointLedger PTUSlabCheckpoints;

  mutable llvm::DenseMap<PTUID, PTUStateInfo> PTUStateInfos;

  // Stores all declaration footprint state in one map instead of maintaining
  // separate maps for each declaration/specialization kind. The generator
  // creates the appropriate node and chain for each Decl.
  mutable LinkedDeclNodeGenerator Generator;
  mutable llvm::DenseMap<const Decl *, DeclLinkedState> LinkedDecls;

  // Tracks hidden mutations for all supported Decl kinds in one shared tracker,
  // so they can be restored correctly during rollback.
  SweepTracker HiddenMutationTracker;

  FunctionExceptionSpecChain FunctionTypeMutations;
  TypeForDeclChain TagdeclInfos;
  // CanonInjectedTSTChain;

  friend class PTUMutationActions;

  SweepTracker &getHiddenMutationTracker() { return HiddenMutationTracker; }

  /// Get-or-create the CXXClassDeclNode backing Canon, acquiring one from
  /// the generator on first use. The only place a CXXClassDeclNode is
  /// created for this map.
  CXXClassDeclNode &recordNodeFor(const CXXRecordDecl *Canon) const {
    DeclLinkedState &Info = LinkedDecls[Canon];
    if (Info.isNull())
      Info = Generator.acquireRecordNode();
    return *cast<CXXClassDeclNode *>(Info);
  }
  VarDeclNode &varNodeFor(const VarDecl *Canon) const {
    DeclLinkedState &Info = LinkedDecls[Canon];
    if (Info.isNull())
      Info = Generator.acquireVarNode();
    return *cast<VarDeclNode *>(Info);
  }
  FunctionDeclNode &functionNodeFor(const FunctionDecl *Canon) const {
    DeclLinkedState &Info = LinkedDecls[Canon];
    if (Info.isNull())
      Info = Generator.acquireFunctionNode();
    return *cast<FunctionDeclNode *>(Info);
  }
  EnumDeclNode &enumNodeFor(const EnumDecl *Canon) const {
    DeclLinkedState &Info = LinkedDecls[Canon];
    if (Info.isNull())
      Info = Generator.acquireEnumNode();
    return *cast<EnumDeclNode *>(Info);
  }

  RecordDeclDefinitionDataChain &chainFor(const CXXRecordDecl *RD) {
    CXXClassDeclNode &Node = recordNodeFor(RD->getCanonicalDecl());
    if (!Node.DefData)
      Node.DefData = Generator.acquireDefDataChain();
    return *Node.DefData;
  }

  SpecializationChain &chainFor(const ClassTemplateSpecializationDecl *Spec) {
    CXXClassDeclNode &Node = recordNodeFor(
        cast<ClassTemplateSpecializationDecl>(Spec->getCanonicalDecl()));
    auto *Chain = Node.Spec.dyn_cast<SpecializationChain *>();
    if (!Chain) {
      Chain = Generator.acquireClassSpecChain();
      Node.Spec = Chain;
    }
    return *Chain;
  }

  VarSpecializationChain &chainFor(const VarTemplateSpecializationDecl *Spec) {
    VarDeclNode &Node = varNodeFor(
        cast<VarTemplateSpecializationDecl>(Spec->getCanonicalDecl()));
    auto *Chain = Node.Spec.dyn_cast<VarSpecializationChain *>();
    if (!Chain) {
      Chain = Generator.acquireVarSpecChain();
      Node.Spec = Chain;
    }
    return *Chain;
  }

  FunctionSpecializationChain &chainFor(const FunctionDecl *FD) {
    FunctionDeclNode &Node = functionNodeFor(FD->getCanonicalDecl());
    auto *Chain = Node.Spec.dyn_cast<FunctionSpecializationChain *>();
    if (!Chain) {
      Chain = Generator.acquireFunctionSpecChain();
      Node.Spec = Chain;
    }
    return *Chain;
  }

  MemberSpecializationChain &memberSpecChainFor(const FunctionDecl *FD) {
    FunctionDeclNode &Node = functionNodeFor(FD->getCanonicalDecl());
    auto *Chain = Node.Spec.dyn_cast<MemberSpecializationChain *>();
    if (!Chain) {
      Chain = Generator.acquireMemberSpecChain();
      Node.Spec = Chain;
    }
    return *Chain;
  }
  MemberSpecializationChain &memberSpecChainFor(const VarDecl *VD) {
    VarDeclNode &Node = varNodeFor(VD->getCanonicalDecl());
    auto *Chain = Node.Spec.dyn_cast<MemberSpecializationChain *>();
    if (!Chain) {
      Chain = Generator.acquireMemberSpecChain();
      Node.Spec = Chain;
    }
    return *Chain;
  }
  MemberSpecializationChain &memberSpecChainFor(const CXXRecordDecl *RD) {
    CXXClassDeclNode &Node = recordNodeFor(RD->getCanonicalDecl());
    auto *Chain = Node.Spec.dyn_cast<MemberSpecializationChain *>();
    if (!Chain) {
      Chain = Generator.acquireMemberSpecChain();
      Node.Spec = Chain;
    }
    return *Chain;
  }
  MemberSpecializationChain &memberSpecChainFor(const EnumDecl *ED) {
    EnumDeclNode &Node = enumNodeFor(ED->getCanonicalDecl());
    if (!Node.MemberSpec)
      Node.MemberSpec = Generator.acquireMemberSpecChain();
    return *Node.MemberSpec;
  }

  RecordDeclDefinitionDataChain *getChainFor(const CXXRecordDecl *RD) const {
    CXXClassDeclNode &Node = recordNodeFor(RD->getCanonicalDecl());
    return Node.DefData;
  }

  SpecializationChain *
  getChainFor(const ClassTemplateSpecializationDecl *Spec) const {
    CXXClassDeclNode &Node = recordNodeFor(
        cast<ClassTemplateSpecializationDecl>(Spec->getCanonicalDecl()));
    return Node.Spec.dyn_cast<SpecializationChain *>();
  }

  VarSpecializationChain *
  getChainFor(const VarTemplateSpecializationDecl *Spec) const {
    VarDeclNode &Node = varNodeFor(
        cast<VarTemplateSpecializationDecl>(Spec->getCanonicalDecl()));
    return Node.Spec.dyn_cast<VarSpecializationChain *>();
  }

  FunctionSpecializationChain *getChainFor(const FunctionDecl *FD) const {
    FunctionDeclNode &Node = functionNodeFor(FD->getCanonicalDecl());
    return Node.Spec.dyn_cast<FunctionSpecializationChain *>();
  }

  MemberSpecializationChain *
  getMemberSpecChainFor(const FunctionDecl *FD) const {
    FunctionDeclNode &Node = functionNodeFor(FD->getCanonicalDecl());
    return Node.Spec.dyn_cast<MemberSpecializationChain *>();
  }
  MemberSpecializationChain *getMemberSpecChainFor(const VarDecl *VD) const {
    VarDeclNode &Node = varNodeFor(VD->getCanonicalDecl());
    return Node.Spec.dyn_cast<MemberSpecializationChain *>();
  }
  MemberSpecializationChain *
  getMemberSpecChainFor(const CXXRecordDecl *RD) const {
    CXXClassDeclNode &Node = recordNodeFor(RD->getCanonicalDecl());
    return Node.Spec.dyn_cast<MemberSpecializationChain *>();
  }
  MemberSpecializationChain *getMemberSpecChainFor(const EnumDecl *ED) const {
    EnumDeclNode &Node = enumNodeFor(ED->getCanonicalDecl());
    return Node.MemberSpec;
  }

public:
  IncrementalStateTracker(ASTContext &Ctx, LangMode M) : Ctx(Ctx), Mode(M) {}

  void beginPTU(llvm::SlabCheckPoint CP) {
    CurID = NextID;
    PTUSlabCheckpoints.recordCheckpoint(CurID, CP);
    PTUStateInfos.try_emplace(CurID, PTUStateInfo{CurID});
    ++NextID;
  }

  PTUCheckpointLedger &getPTUSlabCheckpoints() { return PTUSlabCheckpoints; }
  bool isFromThisPTU(const void *Ptr, PTUID ID) {
    return PTUSlabCheckpoints.isFromThisPTU(Ctx, Ptr, ID);
  }

  PTUStateInfo &current() const {
    assert(NextID > 0 && "no PTU has been started");
    auto It = PTUStateInfos.find(CurID);
    assert(It != PTUStateInfos.end());
    return It->second;
  }

  // Pops this PTU's own tail entry (if any) from *Field. If the chain is
  // now empty -- meaning everything it ever held belonged to the PTU being
  // rolled back -- releases it back to its pool and nulls the pointer.
  // Returns whether the chain still has surviving (earlier-PTU) history.
  template <typename ChainT>
  static bool rollbackChainField(ChainT *&Field, PTUID ID,
                                 LinkedDeclNodeGenerator &Gen);

  static bool rollbackRecordSpec(
      llvm::PointerUnion<SpecializationChain *, MemberSpecializationChain *>
          &Spec,
      PTUID ID, LinkedDeclNodeGenerator &Gen);
  void removeLinkedDecl(const Decl *D, PTUID ID);
  // just only remove entries from current();
  void undoLastEntries();
};

class PTUMutationActions {
private:
  IncrementalStateTracker &Tracker;

public:
  explicit PTUMutationActions(IncrementalStateTracker &Tracker)
      : Tracker(Tracker) {}

  /// Every mutation this PTU recorded for a class-shaped decl, applied in
  /// dependency order: definition data first (later steps read the completed
  /// definition), then the type cache, then specialization footprints, then
  /// lazily-completed members last (they can add members that the steps above
  /// would otherwise have missed).
  ///
  /// Kinds are a bitmask, not alternatives -- one class can legitimately have
  /// several set in a single PTU (e.g. a specialization that also had its
  /// definition data completed), so these are sequential checks, not a switch.
  void commitClassFamily(PTUID ID, const CXXRecordDecl *RD, MutationRecord &Rec,
                         bool IsNew);

  /// Every mutation this PTU recorded for a function-shaped decl, or -- when
  /// CR is non-null -- the baseline seed for one this PTU created.
  ///
  /// Kinds are a bitmask, not alternatives: one function can have its
  /// exception spec resolved AND its body instantiated in the same PTU, so
  /// these are sequential checks rather than a switch.
  ///
  /// Order matters: type-affecting mutations (exception spec, deduced return)
  /// come first because the specialization footprint below reads the
  /// function's type; body instantiation comes last because it can only
  /// happen once everything about the signature is settled.
  void commitFunctionFamily(PTUID ID, const FunctionDecl *FD,
                            MutationRecord &Rec, bool IsNew);

  /// Every mutation this PTU recorded for a var-shaped decl, or -- when CR is
  /// non-null -- the baseline seed for one this PTU created.
  ///
  /// Ordering: initializer instantiation first (it produces the expression
  /// that constant evaluation later consumes), then the cached evaluated
  /// value, then specialization/member footprints which read both.
  void commitVarFamily(PTUID ID, const VarDecl *VD, MutationRecord &Rec,
                       bool IsNew);

  /// Every mutation this PTU recorded for an enum-shaped decl, or -- when CR
  /// is non-null -- the baseline seed for one this PTU created.
  ///
  /// The smallest family: enums have no specialization category (there is no
  /// such thing as an enum template), no deferred bodies, and no members with
  /// independent mutable state -- EnumConstantDecls live and die with the
  /// EnumDecl, so they are not separately tracked.
  void commitEnumFamily(PTUID ID, const EnumDecl *ED, MutationRecord &Rec,
                        bool IsNew);

  // static DeclShape classifyShape(const Decl *D);

  // class DeclStateCommitProxy;

  // class DeclStateRestoreProxy;

  template <typename DeclStateProxyT>
  void walkDecls(const DeclContext *DC, DeclStateProxyT &Proxy);

  // static uint32_t classifyPossibleKinds(DeclShape S, const Decl *D);

  /// Given a Decl already known to be DeclShape S with FlaggedKinds set
  /// (more than one bit at once is the normal case, not an edge case --
  /// a single FunctionDecl can be MSI-backed AND have an unresolved
  /// exception spec at the same time), returns the subset of
  /// FlaggedKinds that actually differ from the last recorded snapshot.
  /// Real chain access, not a guess: builds a fresh footprint the same
  /// way commit() does and compares it against chainFor/
  /// memberSpecChainFor's mostRecent() via the same operator== every
  /// StateAwareChain already uses for its own dedup. A bit with no
  /// chain anywhere in this file (DeducedReturnType has none -- see its
  /// own listener override's comment; TypeForDeclChanged/
  /// EvaluatedValueCached likewise) is never returned as verified --
  /// there is nothing to compare it against, so it cannot be confirmed,
  /// full stop, not assumed either way.
  // static uint32_t verifyMutationFor(const Decl *D, DeclShape S, uint32_t FlaggedKinds,
  //                            PTUID ID);

  void commitLevel1(PTUID ID, const Decl *D, MutationRecord &Rec,
                    bool IsNew = false);

  // respect the LIFO so only current inside map not commited can be commited
  // not randon PTUID
  // global map info shouldn't be commited before only added here. not note*
  // time.
  void commit(TranslationUnitDecl *MostRecentTU);

  void restoreClassFamily(PTUID ID, const CXXRecordDecl *RD,
                          MutationRecord &Rec);

  void restoreFunctionFamily(PTUID ID, const FunctionDecl *FD,
                             MutationRecord &Rec);

  void restoreTemplateFamily(PTUID ID, const RedeclarableTemplateDecl *TD,
                             MutationRecord &Rec);

  void restoreVarFamily(PTUID ID, const VarDecl *VD, MutationRecord &Rec);

  /// Every mutation this PTU recorded for an enum-shaped decl, or -- when CR
  /// is non-null -- the baseline seed for one this PTU created.
  ///
  /// The smallest family: enums have no specialization category (there is no
  /// such thing as an enum template), no deferred bodies, and no members with
  /// independent mutable state -- EnumConstantDecls live and die with the
  /// EnumDecl, so they are not separately tracked.
  void restoreEnumFamily(PTUID ID, const EnumDecl *ED, MutationRecord &Rec);

  void restoreLevel1(PTUID ID, const Decl *D, MutationRecord &Rec);

  void rollback(TranslationUnitDecl *MostRecentTU);
};

class PTUMutationRecorder : public ASTMutationListener {
private:
  IncrementalStateTracker &Tracker;

  template <typename TemplateT, typename SpecT>
  void noteTemplateDeclMutation(const TemplateT *TD, const SpecT *Spec,
                                DeclShape S) {}

  void noteExceptionSpecMutation(const FunctionDecl *FD,
                                 MutationType K = MutationType::ExceptionSpec) {
  }

  void noteDefinitionInstantiated(const Decl *D) {}

public:
  PTUMutationRecorder(IncrementalStateTracker &Tracker) : Tracker(Tracker) {}

  /// A new TagDecl definition was completed.
  void CompletedTagDefinition(const TagDecl *D) override {}

  /// A new declaration with name has been added to a DeclContext.
  void AddedVisibleDecl(const DeclContext *DC, const Decl *D) override {}

  /// An implicit member was added after the definition was completed.
  void AddedCXXImplicitMember(const CXXRecordDecl *RD, const Decl *D) override {
  }

  /// A template specialization (or partial one) was added to the
  /// template declaration.
  void AddedCXXTemplateSpecialization(
      const ClassTemplateDecl *TD,
      const ClassTemplateSpecializationDecl *D) override {}

  /// A template specialization (or partial one) was added to the
  /// template declaration.
  void AddedCXXTemplateSpecialization(
      const VarTemplateDecl *TD,
      const VarTemplateSpecializationDecl *D) override {}

  /// A template specialization (or partial one) was added to the
  /// template declaration.
  void AddedCXXTemplateSpecialization(const FunctionTemplateDecl *TD,
                                      const FunctionDecl *D) override {}

  /// A function's exception specification has been evaluated or
  /// instantiated.
  void ResolvedExceptionSpec(const FunctionDecl *FD) override {}

  /// A function's return type has been deduced.
  void DeducedReturnType(const FunctionDecl *FD, QualType ReturnType) override {
  }

  /// A virtual destructor's operator delete has been resolved.
  void ResolvedOperatorDelete(const CXXDestructorDecl *DD,
                              const FunctionDecl *Delete,
                              Expr *ThisArg) override {}

  /// A virtual destructor's operator global delete has been resolved.
  void ResolvedOperatorGlobDelete(const CXXDestructorDecl *DD,
                                  const FunctionDecl *GlobDelete) override {}

  /// A virtual destructor's operator array delete has been resolved.
  void ResolvedOperatorArrayDelete(const CXXDestructorDecl *DD,
                                   const FunctionDecl *ArrayDelete) override {}

  /// A virtual destructor's operator global array delete has been resolved.
  void ResolvedOperatorGlobArrayDelete(
      const CXXDestructorDecl *DD,
      const FunctionDecl *GlobArrayDelete) override {}

  /// An implicit member got a definition.
  void CompletedImplicitDefinition(const FunctionDecl *D) override {
    noteDefinitionInstantiated(D);
  }

  /// The instantiation of a templated function or variable was
  /// requested. In particular, the point of instantiation and template
  /// specialization kind of \p D may have changed.
  void InstantiationRequested(const ValueDecl *D) override {}

  /// A templated variable's definition was implicitly instantiated.
  void VariableDefinitionInstantiated(const VarDecl *D) override {
    noteDefinitionInstantiated(D);
    // handle Memberinfo
  }

  /// A function template's definition was instantiated.
  void FunctionDefinitionInstantiated(const FunctionDecl *D) override {
    noteDefinitionInstantiated(D);
    // handle Memberinfo
  }

  /// A default argument was instantiated.
  void DefaultArgumentInstantiated(const ParmVarDecl *D) override {}

  /// A default member initializer was instantiated.
  void DefaultMemberInitializerInstantiated(const FieldDecl *D) override {}

  /// A new objc category class was added for an interface.
  void AddedObjCCategoryToInterface(const ObjCCategoryDecl *CatD,
                                    const ObjCInterfaceDecl *IFD) override {}

  /// A declaration is marked used which was not previously marked used.
  ///
  /// \param D the declaration marked used
  void DeclarationMarkedUsed(const Decl *D) override {}

  /// A declaration is marked as OpenMP threadprivate which was not
  /// previously marked as threadprivate.
  ///
  /// \param D the declaration marked OpenMP threadprivate.
  void DeclarationMarkedOpenMPThreadPrivate(const Decl *D) override {}

  /// A declaration is marked as OpenMP groupprivate which was not
  /// previously marked as groupprivate.
  ///
  /// \param D the declaration marked OpenMP groupprivate.
  void DeclarationMarkedOpenMPGroupPrivate(const Decl *D) override {}

  /// A declaration is marked as OpenMP declaretarget which was not
  /// previously marked as declaretarget.
  ///
  /// \param D the declaration marked OpenMP declaretarget.
  /// \param Attr the added attribute.
  void DeclarationMarkedOpenMPDeclareTarget(const Decl *D,
                                            const Attr *Attr) override {}

  /// A declaration is marked as a variable with OpenMP allocator.
  ///
  /// \param D the declaration marked as a variable with OpenMP allocator.
  void DeclarationMarkedOpenMPAllocate(const Decl *D, const Attr *A) override {}

  /// A declaration is marked as an OpenMP indirect call target.
  ///
  /// \param D the declaration marked as an indirect call target.
  void DeclarationMarkedOpenMPIndirectCall(const Decl *D) override {}

  /// A definition has been made visible by being redefined locally.
  ///
  /// \param D The definition that was previously not visible.
  /// \param M The containing module in which the definition was made visible,
  ///        if any.
  void RedefinedHiddenDefinition(const NamedDecl *D, Module *M) override {}

  /// An attribute was added to a RecordDecl
  ///
  /// \param Attr The attribute that was added to the Record
  ///
  /// \param Record The RecordDecl that got a new attribute
  void AddedAttributeToRecord(const Attr *Attr,
                              const RecordDecl *Record) override {}

  /// An mangling number was added to a Decl
  ///
  /// \param D The decl that got a mangling number
  ///
  /// \param Number The mangling number that was added to the Decl
  void AddedManglingNumber(const Decl *D, unsigned Number) override {}

  /// An static local number was added to a Decl
  ///
  /// \param D The decl that got a static local number
  ///
  /// \param Number The static local number that was added to the Decl
  void AddedStaticLocalNumbers(const Decl *D, unsigned Number) override {}

  /// An anonymous namespace was added the translation unit decl
  ///
  /// \param TU The translation unit decl that got a new anonymous namespace
  ///
  /// \param AnonNamespace The anonymous namespace that was added
  void AddedAnonymousNamespace(const TranslationUnitDecl *TU,
                               NamespaceDecl *AnonNamespace) override {}

  void AddedTagDeclType(const TagDecl *TD, const Type *T) override {}
};
} // end namespace clang
#endif // LLVM_CLANG_INTERPRETER_ERROR_RECOVERY_H
