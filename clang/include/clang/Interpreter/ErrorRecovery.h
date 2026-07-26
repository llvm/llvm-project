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

#include "clang/Sema/Sema.h"

namespace clang {
class ASTContext;
class Sema;

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
  size_t TemplateInstCallbacksSize = 0;
  size_t PendingInstantiationsSize = 0;
  size_t LateParsedInstantiationsSize = 0;
  size_t SavedVTableUsesSize = 0;
  size_t SavedPendingInstantiationsSize = 0;
  size_t PendingLocalImplicitInstantiationsSize = 0;
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
///   SemaStateStash Stash(S);
///   Stash.stash(CP);
///   // ... parse ...
///   if (failed)
///     Stash.restore(CP, ASTSlabCP);
class SemaStateStash {
  Sema &S;

  /// Holds full copies of PragmaStack, PragmaClangSection, FileNullabilityMap,
  /// etc.
  //   struct PragmaSnapshot;
  //   std::unique_ptr<PragmaSnapshot> Pragmas;

public:
  explicit SemaStateStash(Sema &S) : S(S) {}
  //   ~SemaStateStash();

  //   SemaStateStash(const SemaStateStash &) = delete;
  //   SemaStateStash &operator=(const SemaStateStash &) = delete;

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

  size_t RequireVectorDeletingDtorSize = 0;

  size_t MergedDeclsSize = 0;
  size_t DeclAttrsSize = 0;
  size_t MergedDefModulesSize = 0;

  size_t ModuleInitializersSize = 0;
  size_t PrimaryModuleNameMapSize = 0;
  size_t SameModuleLookupSetSize = 0;

  size_t ScalableVecTyMapSize = 0;
  size_t LambdaCastPathsSize = 0;
  size_t DeclRawCommentsSize = 0;
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
};

class ASTContextStateStash {
private:
  ASTContext &Ctx;

public:
  explicit ASTContextStateStash(ASTContext &Ctx) : Ctx(Ctx) {}

  ASTContextStateStash(const ASTContextStateStash &) = delete;
  ASTContextStateStash &operator=(const ASTContextStateStash &) = delete;

  void stash(StashCheckPoint &CP);
  void restore(StashCheckPoint &CP, llvm::SlabCheckPoint SlabCP);
  void commit();
};
} // end namespace clang
#endif // LLVM_CLANG_INTERPRETER_ERROR_RECOVERY_H
