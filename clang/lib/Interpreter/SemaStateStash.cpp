//===--- SemaStateStash.cpp - Sema persistent state stash/restore
//----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/Interpreter/ErrorRecovery.h"

#include "clang/AST/ASTContext.h"
#include "clang/AST/Decl.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/DeclTemplate.h"
#include "clang/AST/Expr.h"
#include "clang/AST/ExprCXX.h"
#include "clang/AST/Type.h"
#include "clang/Basic/IdentifierTable.h"
#include "clang/Sema/Sema.h"
#include "clang/Sema/SemaInternal.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/FoldingSet.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/SmallVector.h"

namespace clang {

// template <typename EntryType, typename PredT>
// static void eraseFoldingSetIf(llvm::FoldingSet<EntryType> &FS, PredT &&Pred)
// {
//   SmallVector<EntryType *, 16> ToRemove;
//   for (auto &N : FS)
//     if (Pred(N))
//       ToRemove.push_back(&N);
//   for (auto *N : ToRemove)
//     FS.RemoveNode(N);
// }

// template <typename EntryType, typename PredT>
// static void eraseContextualFoldingSetIf(
//     llvm::ContextualFoldingSet<EntryType, const ASTContext &> &FS,
//     PredT &&Pred) {
//   SmallVector<EntryType *, 16> ToRemove;
//   for (auto &N : FS)
//     if (Pred(N))
//       ToRemove.push_back(&N);
//   for (auto *N : ToRemove)
//     FS.RemoveNode(N);
// }

/// Erase from DenseMap based on predicate
template <typename KeyT, typename ValueT, typename PredT>
static void eraseDenseMapIf(llvm::DenseMap<KeyT, ValueT> &Map, PredT &&Pred) {
  SmallVector<KeyT, 16> ToRemove;

  for (auto &KV : Map)
    if (Pred(KV))
      ToRemove.push_back(KV.getFirst());

  for (auto &Key : ToRemove)
    Map.erase(Key);
}

// template <typename ValueT, typename PredT>
// static void eraseDenseSetIf(llvm::DenseSet<ValueT> &Set, PredT &&Pred) {
//   SmallVector<ValueT, 16> ToRemove;
//   for (const auto &Val : Set)
//     if (Pred(Val))
//       ToRemove.push_back(Val);
//   for (const auto &Val : ToRemove)
//     Set.erase(Val);
// }

// template <typename T, typename PredT>
// static void eraseSmallPtrSetIf(llvm::SmallPtrSet<T, 4> &Set, PredT &&Pred) {
//   SmallVector<T, 8> ToRemove;
//   for (T Val : Set)
//     if (Pred(Val))
//       ToRemove.push_back(Val);
//   for (T Val : ToRemove)
//     Set.erase(Val);
// }

// template <typename T, unsigned N, typename PredT>
// static void eraseSmallSetVectorIf(llvm::SmallSetVector<T, N> &SV,
//                                   PredT &&Pred) {
//   SmallVector<T, 8> ToRemove;
//   for (const T &Val : SV)
//     if (Pred(Val))
//       ToRemove.push_back(Val);
//   for (const T &Val : ToRemove)
//     SV.remove(Val);
// }

// template <typename KeyT, typename ValueT, typename PredT>
// static void eraseMapVectorIf(llvm::MapVector<KeyT, ValueT> &MV, PredT &&Pred)
// {
//   SmallVector<KeyT, 16> ToRemove;
//   for (auto &KV : MV)
//     if (Pred(KV))
//       ToRemove.push_back(KV.first);
//   for (const auto &Key : ToRemove)
//     MV.erase(Key);
// }

// template <typename T, typename PredT>
// static void eraseVectorIf(SmallVectorImpl<T> &Vec, PredT &&Pred) {
//   llvm::erase_if(Vec, std::forward<PredT>(Pred));
// }

//===----------------------------------------------------------------------===//
// Pragma / value snapshot (uses Sema friend access for nested types)
//===----------------------------------------------------------------------===//

// struct SemaStateStash::PragmaSnapshot {
//   Sema::PragmaClangSection PragmaClangBSSSection;
//   Sema::PragmaClangSection PragmaClangDataSection;
//   Sema::PragmaClangSection PragmaClangRodataSection;
//   Sema::PragmaClangSection PragmaClangRelroSection;
//   Sema::PragmaClangSection PragmaClangTextSection;

//   Sema::PragmaStack<MSVtorDispMode> VtorDispStack;
//   Sema::PragmaStack<Sema::AlignPackInfo> AlignPackStack;
//   Sema::PragmaStack<StringLiteral *> DataSegStack;
//   Sema::PragmaStack<StringLiteral *> BSSSegStack;
//   Sema::PragmaStack<StringLiteral *> ConstSegStack;
//   Sema::PragmaStack<StringLiteral *> CodeSegStack;
//   Sema::PragmaStack<bool> StrictGuardStackCheckStack;
//   Sema::PragmaStack<FPOptionsOverride> FpPragmaStack;

//   StringLiteral *CurInitSeg = nullptr;
//   SourceLocation CurInitSegLoc;
//   bool MSPragmaOptimizeIsOn = true;
//   SourceLocation OptimizeOffPragmaLocation;

//   FileNullabilityMap NullabilityMap;
// };

void SemaStateStash::stash(SemaStashCheckPoint &CP) {
  CP.SemaBumpSlabCP = S.BumpAlloc.checkPoint();
  // CP.CachedFunctionScopeSize = S.CachedFunctionScope.size();
  CP.FunctionScopesSize = S.FunctionScopes.size();

  // CP.Ident_superSize = S.Ident_super.size();

  /// --- Pragma ---
  // CP.PragmaClangBSSSectionSize = S.PragmaClangBSSSection.size();
  // CP.PragmaClangDataSectionSize = S.PragmaClangDataSection.size();
  // CP.PragmaClangRodataSectionSize = S.PragmaClangRodataSection.size();
  // CP.PragmaClangRelroSectionSize = S.PragmaClangRelroSection.size();
  // CP.PragmaClangTextSectionSize = S.PragmaClangTextSection.size();
  // CP.VtorDispStackSize = S.VtorDispStack.size();
  // CP.AlignPackStackSize = S.AlignPackStack.size();
  // CP.AlignPackIncludeStackSize = S.AlignPackIncludeStack.size();
  // CP.DataSegStackSize = S.DataSegStack.size();
  // CP.BSSSegStackSize = S.BSSSegStack.size();
  // CP.ConstSegStackSize = S.ConstSegStack.size();
  // CP.CodeSegStackSize = S.CodeSegStack.size();
  // CP.StrictGuardStackCheckStackSize = S.StrictGuardStackCheckStack.size();
  // CP.FpPragmaStackSize = S.FpPragmaStack.size();

  // CP.FunctionToSectionMapSize = S.FunctionToSectionMap.size();
  // CP.PragmaAttributeStackSize = S.PragmaAttributeStack.size();
  // CP.MSFunctionNoBuiltinsSize = S.MSFunctionNoBuiltins.size();
  // CP.PendingExportedNamesSize = S.PendingExportedNames.size();
  CP.TypoCorrectedFunctionDefinitionsSize =
      S.TypoCorrectedFunctionDefinitions.size();
  CP.FlagBitsCacheSize = S.FlagBitsCache.size();
  CP.AssignEnumCacheSize = S.AssignEnumCache.size();
  CP.WeakUndeclaredIdentifiersSize = S.WeakUndeclaredIdentifiers.size();
  CP.ExtnameUndeclaredIdentifiersSize = S.ExtnameUndeclaredIdentifiers.size();
  CP.UnusedLocalTypedefNameCandidatesSize =
      S.UnusedLocalTypedefNameCandidates.size();
  CP.UnusedFileScopedDeclsSize = S.UnusedFileScopedDecls.end();
  CP.TentativeDefinitionsSize = S.TentativeDefinitions.end();
  CP.ExternalDeclarationsSize = S.ExternalDeclarations.size();
  CP.ParsingInitForAutoVarsSize = S.ParsingInitForAutoVars.size();
  CP.DeclsToCheckForDeferredDiagsSize = S.DeclsToCheckForDeferredDiags.size();
  CP.ShadowingDeclsSize = S.ShadowingDecls.size();
  CP.WeakTopLevelDeclSize = S.WeakTopLevelDecl.size();
  CP.ExtVectorDeclsSize = S.ExtVectorDecls.end();
  CP.VTableUsesSize = S.VTableUses.size();
  CP.VTablesUsedSize = S.VTablesUsed.size();
  CP.DelayedDllExportClassesSize = S.DelayedDllExportClasses.size();
  CP.DelayedDllExportMemberFunctionsSize =
      S.DelayedDllExportMemberFunctions.size();
  CP.InventedParameterInfosSize = S.InventedParameterInfos.size();
  // CP.FieldCollectorSize = S.FieldCollector.size();
  CP.UnusedPrivateFieldsSize = S.UnusedPrivateFields.size();
  if (S.PureVirtualClassDiagSet)
    CP.PureVirtualClassDiagSetSize = S.PureVirtualClassDiagSet->size();
  CP.DelegatingCtorDeclsSize = S.DelegatingCtorDecls.end();
  // CP.StdNamespaceSize = S.StdNamespace.size();
  CP.UnparsedDefaultArgLocsSize = S.UnparsedDefaultArgLocs.size();
  CP.UndefinedButUsedSize = S.UndefinedButUsed.size();
  CP.SpecialMembersBeingDeclaredSize = S.SpecialMembersBeingDeclared.size();
  CP.DelayedOverridingExceptionSpecChecksSize =
      S.DelayedOverridingExceptionSpecChecks.size();
  CP.DelayedEquivalentExceptionSpecChecksSize =
      S.DelayedEquivalentExceptionSpecChecks.size();
  CP.MaybeODRUseExprsSize = S.MaybeODRUseExprs.size();
  CP.RefsMinusAssignmentsSize = S.RefsMinusAssignments.size();
  CP.ExprCleanupObjectsSize = S.ExprCleanupObjects.size();
  CP.ExprEvalContextsSize = S.ExprEvalContexts.size();
  CP.FailedImmediateInvocationsSize = S.FailedImmediateInvocations.size();
  // CP.ImplicitlyRetainedSelfLocsSize = S.ImplicitlyRetainedSelfLocs.size();
  CP.DeleteExprsSize = S.DeleteExprs.size();
  CP.CurrentParameterCopyTypesSize = S.CurrentParameterCopyTypes.size();
  CP.AggregateDeductionCandidatesSize = S.AggregateDeductionCandidates.size();
  CP.TypoCorrectionFailuresSize = S.TypoCorrectionFailures.size();
  CP.SpecialMemberCacheSize = S.SpecialMemberCache.size();
  CP.ModuleScopesSize = S.ModuleScopes.size();
  CP.DeferredExportedNamespacesSize = S.DeferredExportedNamespaces.size();
  CP.PendingInlineFuncDeclsSize = S.PendingInlineFuncDecls.size();
  CP.CurrentSEHFinallySize = S.CurrentSEHFinally.size();
  CP.CurrentDeferSize = S.CurrentDefer.size();
  CP.LateParsedTemplateMapSize = S.LateParsedTemplateMap.size();
  CP.SuppressedDiagnosticsSize = S.SuppressedDiagnostics.size();
  // CP.CurrentInstantiationScopeSize = S.CurrentInstantiationScope.size();
  CP.UnparsedDefaultArgInstantiationsSize =
      S.UnparsedDefaultArgInstantiations.size();
  CP.CodeSynthesisContextsSize = S.CodeSynthesisContexts.size();
  CP.InstantiatingSpecializationsSize = S.InstantiatingSpecializations.size();
  CP.InstantiatedNonDependentTypesSize = S.InstantiatedNonDependentTypes.size();
  CP.CodeSynthesisContextLookupModulesSize =
      S.CodeSynthesisContextLookupModules.size();
  CP.LookupModulesCacheSize = S.LookupModulesCache.size();
  CP.VisibleNamespaceCacheSize = S.VisibleNamespaceCache.size();
  CP.TemplateInstCallbacksSize = S.TemplateInstCallbacks.size();
  CP.PendingInstantiationsSize = S.PendingInstantiations.size();
  CP.LateParsedInstantiationsSize = S.LateParsedInstantiations.size();
  CP.SavedVTableUsesSize = S.SavedVTableUses.size();
  CP.SavedPendingInstantiationsSize = S.SavedPendingInstantiations.size();
  CP.PendingLocalImplicitInstantiationsSize =
      S.PendingLocalImplicitInstantiations.size();
  CP.UnsubstitutedConstraintSatisfactionCacheSize =
      S.UnsubstitutedConstraintSatisfactionCache.size();
  CP.SubsumptionCacheSize = S.SubsumptionCache.size();
  CP.NormalizationCacheSize = S.NormalizationCache.size();
  CP.SatisfactionCacheSize = S.SatisfactionCache.size();
  CP.SatisfactionStackSize = S.SatisfactionStack.size();
  // CP.NullabilityMapSize = S.NullabilityMap.size();
  CP.DeclsWithEffectsToVerifySize = S.DeclsWithEffectsToVerify.size();
  // CP.AllEffectsToVerifySize = S.AllEffectsToVerify.size();
}

void SemaStateStash::restore(SemaStashCheckPoint &CP,
                             llvm::SlabCheckPoint SlabCP) {

  ASTContext &Ctx = S.getASTContext();
  if (CP.FunctionScopesSize != S.FunctionScopes.size()) {
    llvm::dbgs() << "CP.FunctionScopesSize != S.FunctionScopes.size()\n";
    S.FunctionScopes.resize(CP.FunctionScopesSize);
    assert(CP.FunctionScopesSize == S.FunctionScopes.size());
  }

  if (CP.TypoCorrectedFunctionDefinitionsSize !=
      S.TypoCorrectedFunctionDefinitions.size()) {
    llvm::dbgs() << "CP.TypoCorrectedFunctionDefinitionsSize != "
                    "S.TypoCorrectedFunctionDefinitions.size()\n";
    // llvm::SmallPtrSet<const NamedDecl *, 4> TypoCorrectedFunctionDefinitions;
    assert(CP.TypoCorrectedFunctionDefinitionsSize ==
           S.TypoCorrectedFunctionDefinitions.size());
  }

  if (CP.FlagBitsCacheSize != S.FlagBitsCache.size()) {
    llvm::dbgs() << "CP.FlagBitsCacheSize != S.FlagBitsCache.size()\n";
    // mutable llvm::DenseMap<const EnumDecl *, llvm::APInt> FlagBitsCache;
    assert(CP.FlagBitsCacheSize == S.FlagBitsCache.size());
  }

  if (CP.AssignEnumCacheSize != S.AssignEnumCache.size()) {
    llvm::dbgs() << "CP.AssignEnumCacheSize != S.AssignEnumCache.size()\n";
    //   llvm::DenseMap<const EnumDecl *, llvm::SmallVector<llvm::APSInt>>
    //   AssignEnumCache;
    assert(CP.AssignEnumCacheSize == S.AssignEnumCache.size());
  }

  if (CP.WeakUndeclaredIdentifiersSize != S.WeakUndeclaredIdentifiers.size()) {
    llvm::dbgs() << "CP.WeakUndeclaredIdentifiersSize != "
                    "S.WeakUndeclaredIdentifiers.size()\n";
    //  llvm::MapVector<
    //   IdentifierInfo *,
    //   llvm::SetVector<
    //       WeakInfo, llvm::SmallVector<WeakInfo, 1u>,
    //       llvm::SmallDenseSet<WeakInfo, 2u,
    //       WeakInfo::DenseMapInfoByAliasOnly>>>
    //   WeakUndeclaredIdentifiers;
    assert(CP.WeakUndeclaredIdentifiersSize ==
           S.WeakUndeclaredIdentifiers.size());
  }

  if (CP.ExtnameUndeclaredIdentifiersSize !=
      S.ExtnameUndeclaredIdentifiers.size()) {
    llvm::dbgs() << "CP.ExtnameUndeclaredIdentifiersSize != "
                    "S.ExtnameUndeclaredIdentifiers.size()\n";
    //   llvm::DenseMap<IdentifierInfo *, AsmLabelAttr *>
    //   ExtnameUndeclaredIdentifiers;
    assert(CP.ExtnameUndeclaredIdentifiersSize ==
           S.ExtnameUndeclaredIdentifiers.size());
  }

  if (CP.UnusedLocalTypedefNameCandidatesSize !=
      S.UnusedLocalTypedefNameCandidates.size()) {
    llvm::dbgs() << "CP.UnusedLocalTypedefNameCandidatesSize != "
                    "S.UnusedLocalTypedefNameCandidates.size()\n";
    //   llvm::SmallSetVector<const TypedefNameDecl *, 4>
    //   UnusedLocalTypedefNameCandidates;
    assert(CP.UnusedLocalTypedefNameCandidatesSize ==
           S.UnusedLocalTypedefNameCandidates.size());
  }

  if (CP.UnusedFileScopedDeclsSize != S.UnusedFileScopedDecls.end()) {
    llvm::dbgs()
        << "CP.UnusedFileScopedDeclsSize != S.UnusedFileScopedDecls.end()\n";
    //  typedef LazyVector<const DeclaratorDecl *, ExternalSemaSource,
    //  &ExternalSemaSource::ReadUnusedFileScopedDecls, 2, 2>
    //  UnusedFileScopedDeclsType;
    assert(CP.UnusedFileScopedDeclsSize == S.UnusedFileScopedDecls.end());
  }

  if (CP.TentativeDefinitionsSize != S.TentativeDefinitions.end()) {
    llvm::dbgs()
        << "CP.TentativeDefinitionsSize != S.TentativeDefinitions.size()\n";
    //    typedef LazyVector<VarDecl *, ExternalSemaSource,
    //                  &ExternalSemaSource::ReadTentativeDefinitions, 2, 2>
    //   TentativeDefinitionsType;
    assert(CP.TentativeDefinitionsSize == S.TentativeDefinitions.end());
  }

  if (CP.ExternalDeclarationsSize != S.ExternalDeclarations.size()) {
    llvm::dbgs()
        << "CP.ExternalDeclarationsSize != S.ExternalDeclarations.size()\n";
    //    SmallVector<DeclaratorDecl *, 4> ExternalDeclarations;
    assert(CP.ExternalDeclarationsSize == S.ExternalDeclarations.size());
  }

  if (CP.ParsingInitForAutoVarsSize != S.ParsingInitForAutoVars.size()) {
    llvm::dbgs()
        << "CP.ParsingInitForAutoVarsSize != S.ParsingInitForAutoVars.size()\n";
    //   llvm::SmallPtrSet<const Decl *, 4> ParsingInitForAutoVars;
    assert(CP.ParsingInitForAutoVarsSize == S.ParsingInitForAutoVars.size());
  }

  if (CP.DeclsToCheckForDeferredDiagsSize !=
      S.DeclsToCheckForDeferredDiags.size()) {
    llvm::dbgs() << "CP.DeclsToCheckForDeferredDiagsSize != "
                    "S.DeclsToCheckForDeferredDiags.size()\n";
    //   llvm::SmallSetVector<Decl *, 4> DeclsToCheckForDeferredDiags;
    assert(CP.DeclsToCheckForDeferredDiagsSize ==
           S.DeclsToCheckForDeferredDiags.size());
  }

  if (CP.ShadowingDeclsSize != S.ShadowingDecls.size()) {
    llvm::dbgs() << "CP.ShadowingDeclsSize != S.ShadowingDecls.size()\n";
    //   llvm::DenseMap<const NamedDecl *, const NamedDecl *> ShadowingDecls;
    assert(CP.ShadowingDeclsSize == S.ShadowingDecls.size());
  }

  if (CP.WeakTopLevelDeclSize != S.WeakTopLevelDecl.size()) {
    llvm::dbgs() << "CP.WeakTopLevelDeclSize != S.WeakTopLevelDecl.size()\n";
    //    SmallVector<Decl *, 2> WeakTopLevelDecl;
    assert(CP.WeakTopLevelDeclSize == S.WeakTopLevelDecl.size());
  }

  if (CP.ExtVectorDeclsSize != S.ExtVectorDecls.end()) {
    llvm::dbgs() << "CP.ExtVectorDeclsSize != S.ExtVectorDecls.size()\n";
    //   typedef LazyVector<TypedefNameDecl *, ExternalSemaSource,
    //  &ExternalSemaSource::ReadExtVectorDecls, 2, 2> ExtVectorDecls;
    assert(CP.ExtVectorDeclsSize == S.ExtVectorDecls.end());
  }

  if (CP.VTableUsesSize != S.VTableUses.size()) {
    llvm::dbgs() << "CP.VTableUsesSize != S.VTableUses.size()\n";
    //   SmallVector<VTableUse, 16> VTableUses;
    assert(CP.VTableUsesSize == S.VTableUses.size());
  }

  if (CP.VTablesUsedSize != S.VTablesUsed.size()) {
    llvm::dbgs() << "CP.VTablesUsedSize != S.VTablesUsed.size()\n";
    //   llvm::DenseMap<CXXRecordDecl *, bool> VTablesUsed;
    assert(CP.VTablesUsedSize == S.VTablesUsed.size());
  }

  if (CP.DelayedDllExportClassesSize != S.DelayedDllExportClasses.size()) {
    llvm::dbgs() << "CP.DelayedDllExportClassesSize != "
                    "S.DelayedDllExportClasses.size()\n";
    //   SmallVector<CXXRecordDecl *, 4> DelayedDllExportClasses;
    assert(CP.DelayedDllExportClassesSize == S.DelayedDllExportClasses.size());
  }

  if (CP.DelayedDllExportMemberFunctionsSize !=
      S.DelayedDllExportMemberFunctions.size()) {
    llvm::dbgs() << "CP.DelayedDllExportMemberFunctionsSize != "
                    "S.DelayedDllExportMemberFunctions.size()\n";
    //   SmallVector<CXXMethodDecl *, 4> DelayedDllExportMemberFunctions;
    assert(CP.DelayedDllExportMemberFunctionsSize ==
           S.DelayedDllExportMemberFunctions.size());
  }

  if (CP.InventedParameterInfosSize != S.InventedParameterInfos.size()) {
    llvm::dbgs()
        << "CP.InventedParameterInfosSize != S.InventedParameterInfos.size()\n";
    //   SmallVector<InventedTemplateParameterInfo, 4> InventedParameterInfos;
    assert(CP.InventedParameterInfosSize == S.InventedParameterInfos.size());
  }

  if (CP.UnusedPrivateFieldsSize != S.UnusedPrivateFields.size()) {
    llvm::dbgs()
        << "CP.UnusedPrivateFieldsSize != S.UnusedPrivateFields.size()\n";
    //   typedef llvm::SmallSetVector<const NamedDecl *, 16> NamedDeclSetType;
    /// Set containing all declared private fields that are not used.
    //   NamedDeclSetType UnusedPrivateFields;
    assert(CP.UnusedPrivateFieldsSize == S.UnusedPrivateFields.size());
  }

  if (S.PureVirtualClassDiagSet &&
      (CP.PureVirtualClassDiagSetSize != S.PureVirtualClassDiagSet->size())) {
    llvm::dbgs() << "CP.PureVirtualClassDiagSetSize != "
                    "S.PureVirtualClassDiagSet.size()\n";
    //   typedef llvm::SmallPtrSet<const CXXRecordDecl *, 8> RecordDeclSetTy;

    /// PureVirtualClassDiagSet - a set of class declarations which we have
    /// emitted a list of pure virtual functions. Used to prevent emitting the
    /// same list more than once.
    //   std::unique_ptr<RecordDeclSetTy> PureVirtualClassDiagSet;
    assert(CP.PureVirtualClassDiagSetSize == S.PureVirtualClassDiagSet->size());
  }

  if (CP.DelegatingCtorDeclsSize != S.DelegatingCtorDecls.end()) {
    llvm::dbgs()
        << "CP.DelegatingCtorDeclsSize != S.DelegatingCtorDecls.size()\n";
    //   typedef LazyVector<CXXConstructorDecl *, ExternalSemaSource,
    //  &ExternalSemaSource::ReadDelegatingConstructors, 2, 2>
    //   DelegatingCtorDeclsType;

    /// All the delegating constructors seen so far in the file, used for
    /// cycle detection at the end of the TU.
    //   DelegatingCtorDeclsType DelegatingCtorDecls;

    assert(CP.DelegatingCtorDeclsSize == S.DelegatingCtorDecls.end());
  }

  if (CP.UnparsedDefaultArgLocsSize != S.UnparsedDefaultArgLocs.size()) {
    llvm::dbgs()
        << "CP.UnparsedDefaultArgLocsSize != S.UnparsedDefaultArgLocs.size()\n";
    //   llvm::DenseMap<ParmVarDecl *, SourceLocation> UnparsedDefaultArgLocs;
    assert(CP.UnparsedDefaultArgLocsSize == S.UnparsedDefaultArgLocs.size());
  }

  if (CP.UndefinedButUsedSize != S.UndefinedButUsed.size()) {
    llvm::dbgs() << "CP.UndefinedButUsedSize != S.UndefinedButUsed.size()\n";
      // llvm::MapVector<NamedDecl *, SourceLocation> UndefinedButUsed;
    while (CP.UndefinedButUsedSize != S.UndefinedButUsed.size())
      S.UndefinedButUsed.pop_back();
    assert(CP.UndefinedButUsedSize == S.UndefinedButUsed.size());
  }

  if (CP.SpecialMembersBeingDeclaredSize !=
      S.SpecialMembersBeingDeclared.size()) {
    llvm::dbgs() << "CP.SpecialMembersBeingDeclaredSize != "
                    "S.SpecialMembersBeingDeclared.size()\n";
    //   llvm::SmallPtrSet<SpecialMemberDecl, 4> SpecialMembersBeingDeclared;
    assert(CP.SpecialMembersBeingDeclaredSize ==
           S.SpecialMembersBeingDeclared.size());
  }

  if (CP.SpecialMembersBeingDeclaredSize !=
      S.SpecialMembersBeingDeclared.size()) {
    llvm::dbgs() << "CP.SpecialMembersBeingDeclaredSize != "
                    "S.SpecialMembersBeingDeclared.size()\n";
    //   llvm::SmallPtrSet<SpecialMemberDecl, 4> SpecialMembersBeingDeclared;
    assert(CP.SpecialMembersBeingDeclaredSize ==
           S.SpecialMembersBeingDeclared.size());
  }

  if (CP.DelayedOverridingExceptionSpecChecksSize !=
      S.DelayedOverridingExceptionSpecChecks.size()) {
    llvm::dbgs() << "CP.DelayedOverridingExceptionSpecChecksSize != "
                    "S.DelayedOverridingExceptionSpecChecks.size()\n";
    //     SmallVector<std::pair<const CXXMethodDecl *, const CXXMethodDecl *>,
    //     2>
    //   DelayedOverridingExceptionSpecChecks;
    assert(CP.DelayedOverridingExceptionSpecChecksSize ==
           S.DelayedOverridingExceptionSpecChecks.size());
  }

  if (CP.DelayedEquivalentExceptionSpecChecksSize !=
      S.DelayedEquivalentExceptionSpecChecks.size()) {
    llvm::dbgs() << "CP.DelayedEquivalentExceptionSpecChecksSize != "
                    "S.DelayedEquivalentExceptionSpecChecks.size()\n";
    //   SmallVector<std::pair<FunctionDecl *, FunctionDecl *>, 2>
    //   DelayedEquivalentExceptionSpecChecks;
    assert(CP.DelayedEquivalentExceptionSpecChecksSize ==
           S.DelayedEquivalentExceptionSpecChecks.size());
  }

  if (CP.MaybeODRUseExprsSize != S.MaybeODRUseExprs.size()) {
    llvm::dbgs() << "CP.MaybeODRUseExprsSize != S.MaybeODRUseExprs.size()\n";
    // S.MaybeODRUseExprs.resize(CP.MaybeODRUseExprsSize);
    assert(CP.MaybeODRUseExprsSize == S.MaybeODRUseExprs.size());
  }

  if (CP.RefsMinusAssignmentsSize != S.RefsMinusAssignments.size()) {
    llvm::dbgs()
        << "CP.RefsMinusAssignmentsSize != S.RefsMinusAssignments.size()\n";
    //     llvm::DenseMap<const VarDecl *, int> RefsMinusAssignments;
    eraseDenseMapIf(
        S.RefsMinusAssignments,
        [&](llvm::detail::DenseMapPair<const VarDecl *, int> &KV) -> bool {
          return Ctx.getAllocator().isAfterCheckpoint(
              static_cast<void *>(const_cast<VarDecl *>(KV.getFirst())),
              SlabCP);
        });
    assert(CP.RefsMinusAssignmentsSize == S.RefsMinusAssignments.size());
  }

  if (CP.ExprCleanupObjectsSize != S.ExprCleanupObjects.size()) {
    llvm::dbgs()
        << "CP.ExprCleanupObjectsSize != S.ExprCleanupObjects.size()\n";
    //   SmallVector<ExprWithCleanups::CleanupObject, 8> ExprCleanupObjects;
    assert(CP.ExprCleanupObjectsSize == S.ExprCleanupObjects.size());
  }

  if (CP.ExprEvalContextsSize != S.ExprEvalContexts.size()) {
    llvm::dbgs() << "CP.ExprEvalContextsSize != S.ExprEvalContexts.size()\n";
    //   SmallVector<ExpressionEvaluationContextRecord, 8> ExprEvalContexts;
    assert(CP.ExprEvalContextsSize == S.ExprEvalContexts.size());
  }

  if (CP.FailedImmediateInvocationsSize !=
      S.FailedImmediateInvocations.size()) {
    llvm::dbgs() << "CP.FailedImmediateInvocationsSize != "
                    "S.FailedImmediateInvocations.size()\n";
    //    llvm::SmallPtrSet<ConstantExpr *, 4> FailedImmediateInvocations;
    assert(CP.FailedImmediateInvocationsSize ==
           S.FailedImmediateInvocations.size());
  }

  if (CP.DeleteExprsSize != S.DeleteExprs.size()) {
    llvm::dbgs() << "CP.DeleteExprsSize != S.DeleteExprs.size()\n";
    //   llvm::MapVector<FieldDecl *, DeleteLocs> DeleteExprs;
    assert(CP.DeleteExprsSize == S.DeleteExprs.size());
  }

  if (CP.CurrentParameterCopyTypesSize != S.CurrentParameterCopyTypes.size()) {
    llvm::dbgs() << "CP.CurrentParameterCopyTypesSize != "
                    "S.CurrentParameterCopyTypes.size()\n";
    //    llvm::SmallVector<QualType, 4> CurrentParameterCopyTypes;
    assert(CP.CurrentParameterCopyTypesSize ==
           S.CurrentParameterCopyTypes.size());
  }

  if (CP.AggregateDeductionCandidatesSize !=
      S.AggregateDeductionCandidates.size()) {
    llvm::dbgs() << "CP.AggregateDeductionCandidatesSize != "
                    "S.AggregateDeductionCandidates.size()\n";
    //   llvm::DenseMap<unsigned, CXXDeductionGuideDecl *>
    //   AggregateDeductionCandidates;
    assert(CP.AggregateDeductionCandidatesSize ==
           S.AggregateDeductionCandidates.size());
  }

  if (CP.TypoCorrectionFailuresSize != S.TypoCorrectionFailures.size()) {
    llvm::dbgs() << "CP.TypoCorrectionFailuresSize != "
                    "S.TypoCorrectionFailures.size()\n";
    // eraseDenseMapIf(
    //     S.TypoCorrectionFailures,
    //     [&](llvm::detail::DenseMapPair<IdentifierInfo *, Sema::SrcLocSet> &KV)
    //         -> bool {
    //           llvm::outs() << "SlabCheckPoint Cur : " << (void *)SlabCP.CurPtr << "\n";
    //           llvm::outs() << "SlabCheckPoint END : " << (void *)SlabCP.End << "\n";
    //           llvm::outs() << "Addr: " << static_cast<void *>(KV.getFirst()) << "\n";
    //       return Ctx.getAllocator().isAfterCheckpoint(
    //           static_cast<void *>(KV.getFirst()), SlabCP);
    //     });
    // assert(CP.TypoCorrectionFailuresSize == S.TypoCorrectionFailures.size());
  }

  if (CP.ModuleScopesSize != S.ModuleScopes.size()) {
    llvm::dbgs() << "CP.ModuleScopesSize != "
                    "S.ModuleScopes.size()\n";
    //  llvm::SmallVector<ModuleScope, 16> ModuleScopes;
    assert(CP.ModuleScopesSize == S.ModuleScopes.size());
  }

  if (CP.DeferredExportedNamespacesSize !=
      S.DeferredExportedNamespaces.size()) {
    llvm::dbgs() << "CP.DeferredExportedNamespacesSize != "
                    "S.DeferredExportedNamespaces.size()\n";
    /// Namespace definitions that we will export when they finish.
    // llvm::SmallPtrSet<const NamespaceDecl *, 8> DeferredExportedNamespaces;
    assert(CP.DeferredExportedNamespacesSize ==
           S.DeferredExportedNamespaces.size());
  }

  if (CP.PendingInlineFuncDeclsSize != S.PendingInlineFuncDecls.size()) {
    llvm::dbgs() << "CP.PendingInlineFuncDeclsSize != "
                    "S.PendingInlineFuncDecls.size()\n";
    ///   /// In a C++ standard module, inline declarations require a definition
    ///   to be
    /// present at the end of a definition domain.  This set holds the decls to
    /// be checked at the end of the TU.
    //   llvm::SmallPtrSet<const FunctionDecl *, 8> PendingInlineFuncDecls;

    assert(CP.PendingInlineFuncDeclsSize == S.PendingInlineFuncDecls.size());
  }

  if (CP.CurrentSEHFinallySize != S.CurrentSEHFinally.size()) {
    llvm::dbgs() << "CP.CurrentSEHFinallySize != "
                    "S.CurrentSEHFinally.size()\n";
    //    /// Stack of active SEH __finally scopes.  Can be empty.
    // SmallVector<Scope *, 2> CurrentSEHFinally;

    assert(CP.CurrentSEHFinallySize == S.CurrentSEHFinally.size());
  }

  if (CP.CurrentDeferSize != S.CurrentDefer.size()) {
    llvm::dbgs() << "CP.CurrentDeferSize != "
                    "S.CurrentDefer.size()\n";
    //    /// Stack of '_Defer' statements that are currently being parsed, as
    //    well
    /// as the locations of their '_Defer' keywords. Can be empty.
    //   SmallVector<std::pair<Scope *, SourceLocation>, 2> CurrentDefer;

    assert(CP.CurrentDeferSize == S.CurrentDefer.size());
  }

  if (CP.LateParsedTemplateMapSize != S.LateParsedTemplateMap.size()) {
    llvm::dbgs() << "CP.LateParsedTemplateMapSize != "
                    "S.LateParsedTemplateMap.size()\n";
    //   typedef llvm::MapVector<const FunctionDecl *,
    //                           std::unique_ptr<LateParsedTemplate>>
    //       LateParsedTemplateMapT;
    //   LateParsedTemplateMapT LateParsedTemplateMap;
    assert(CP.LateParsedTemplateMapSize == S.LateParsedTemplateMap.size());
  }

  if (CP.SuppressedDiagnosticsSize != S.SuppressedDiagnostics.size()) {
    llvm::dbgs() << "CP.SuppressedDiagnosticsSize != "
                    "S.SuppressedDiagnostics.size()\n";
    //    typedef llvm::DenseMap<Decl *, SmallVector<PartialDiagnosticAt, 1>>
    //       SuppressedDiagnosticsMap;
    //   SuppressedDiagnosticsMap SuppressedDiagnostics;
    assert(CP.SuppressedDiagnosticsSize == S.SuppressedDiagnostics.size());
  }

  if (CP.UnparsedDefaultArgInstantiationsSize !=
      S.UnparsedDefaultArgInstantiations.size()) {
    llvm::dbgs() << "CP.UnparsedDefaultArgInstantiationsSize != "
                    "S.UnparsedDefaultArgInstantiations.size()\n";

    //   typedef llvm::DenseMap<ParmVarDecl *, llvm::TinyPtrVector<ParmVarDecl
    //   *>>
    //       UnparsedDefaultArgInstantiationsMap;

    //   /// A mapping from parameters with unparsed default arguments to the
    //   /// set of instantiations of each parameter.
    //   ///
    //   /// This mapping is a temporary data structure used when parsing
    //   /// nested class templates or nested classes of class templates,
    //   /// where we might end up instantiating an inner class before the
    //   /// default arguments of its methods have been parsed.
    //   UnparsedDefaultArgInstantiationsMap UnparsedDefaultArgInstantiations;

    assert(CP.UnparsedDefaultArgInstantiationsSize ==
           S.UnparsedDefaultArgInstantiations.size());
  }

  if (CP.CodeSynthesisContextsSize != S.CodeSynthesisContexts.size()) {
    llvm::dbgs() << "CP.CodeSynthesisContextsSize != "
                    "S.CodeSynthesisContexts.size()\n";
    //  SmallVector<CodeSynthesisContext, 16> CodeSynthesisContexts;

    assert(CP.CodeSynthesisContextsSize == S.CodeSynthesisContexts.size());
  }

  if (CP.InstantiatingSpecializationsSize !=
      S.InstantiatingSpecializations.size()) {
    llvm::dbgs() << "CP.InstantiatingSpecializationsSize != "
                    "S.InstantiatingSpecializations.size()\n";
    //    /// Specializations whose definitions are currently being
    //    instantiated.
    //   llvm::DenseSet<InstantiatingSpecializationsKey>
    //   InstantiatingSpecializations;

    assert(CP.InstantiatingSpecializationsSize ==
           S.InstantiatingSpecializations.size());
  }

  if (CP.InstantiatedNonDependentTypesSize !=
      S.InstantiatedNonDependentTypes.size()) {
    llvm::dbgs() << "CP.InstantiatedNonDependentTypesSize != "
                    "S.InstantiatedNonDependentTypes.size()\n";
    //      llvm::DenseSet<QualType> InstantiatedNonDependentTypes;
    assert(CP.InstantiatedNonDependentTypesSize ==
           S.InstantiatedNonDependentTypes.size());
  }

  if (CP.CodeSynthesisContextLookupModulesSize !=
      S.CodeSynthesisContextLookupModules.size()) {
    llvm::dbgs() << "CP.CodeSynthesisContextLookupModulesSize != "
                    "S.CodeSynthesisContextLookupModules.size()\n";
    //      SmallVector<Module *, 16> CodeSynthesisContextLookupModules;

    assert(CP.CodeSynthesisContextLookupModulesSize ==
           S.CodeSynthesisContextLookupModules.size());
  }

  if (CP.LookupModulesCacheSize != S.LookupModulesCache.size()) {
    llvm::dbgs() << "CP.LookupModulesCacheSize != "
                    "S.LookupModulesCache.size()\n";
    //        llvm::DenseSet<Module *> LookupModulesCache;

    assert(CP.LookupModulesCacheSize == S.LookupModulesCache.size());
  }

  if (CP.VisibleNamespaceCacheSize != S.VisibleNamespaceCache.size()) {
    llvm::dbgs() << "CP.VisibleNamespaceCacheSize != "
                    "S.VisibleNamespaceCache.size()\n";
    //           llvm::DenseMap<NamedDecl *, NamedDecl *> VisibleNamespaceCache;

    assert(CP.VisibleNamespaceCacheSize == S.VisibleNamespaceCache.size());
  }

  if (CP.TemplateInstCallbacksSize != S.TemplateInstCallbacks.size()) {
    llvm::dbgs() << "CP.TemplateInstCallbacksSize != "
                    "S.TemplateInstCallbacks.size()\n";
    //    std::vector<std::unique_ptr<TemplateInstantiationCallback>>
    //   TemplateInstCallbacks;
    assert(CP.TemplateInstCallbacksSize == S.TemplateInstCallbacks.size());
  }

  if (CP.PendingInstantiationsSize != S.PendingInstantiations.size()) {
    llvm::dbgs() << "CP.PendingInstantiationsSize != "
                    "S.PendingInstantiations.size()\n";
    //      std::deque<PendingImplicitInstantiation> PendingInstantiations;

    assert(CP.PendingInstantiationsSize == S.PendingInstantiations.size());
  }

  if (CP.LateParsedInstantiationsSize != S.LateParsedInstantiations.size()) {
    llvm::dbgs() << "CP.LateParsedInstantiationsSize != "
                    "S.LateParsedInstantiations.size()\n";
    S.LateParsedInstantiations.resize(CP.LateParsedInstantiationsSize);
    assert(CP.LateParsedInstantiationsSize ==
           S.LateParsedInstantiations.size());
  }

  if (CP.SavedVTableUsesSize != S.SavedVTableUses.size()) {
    llvm::dbgs() << "CP.SavedVTableUsesSize != "
                    "S.SavedVTableUses.size()\n";
    S.SavedVTableUses.resize(CP.SavedVTableUsesSize);
    assert(CP.SavedVTableUsesSize == S.SavedVTableUses.size());
  }

  if (CP.SavedPendingInstantiationsSize !=
      S.SavedPendingInstantiations.size()) {
    llvm::dbgs() << "CP.SavedPendingInstantiationsSize != "
                    "S.SavedPendingInstantiations.size()\n";
    S.SavedPendingInstantiations.resize(CP.SavedPendingInstantiationsSize);
    assert(CP.SavedPendingInstantiationsSize ==
           S.SavedPendingInstantiations.size());
  }

  if (CP.PendingLocalImplicitInstantiationsSize !=
      S.PendingLocalImplicitInstantiations.size()) {
    llvm::dbgs() << "CP.PendingLocalImplicitInstantiationsSize != "
                    "S.PendingLocalImplicitInstantiations.size()\n";
    S.PendingLocalImplicitInstantiations.resize(
        CP.PendingLocalImplicitInstantiationsSize);
    assert(CP.PendingLocalImplicitInstantiationsSize ==
           S.PendingLocalImplicitInstantiations.size());
  }

  if (CP.UnsubstitutedConstraintSatisfactionCacheSize !=
      S.UnsubstitutedConstraintSatisfactionCache.size()) {
    llvm::dbgs() << "CP.UnsubstitutedConstraintSatisfactionCacheSize != "
                    "S.UnsubstitutedConstraintSatisfactionCache.size()\n";
    // eraseDenseMapIf(
    //     S.UnsubstitutedConstraintSatisfactionCache,
    //     [&](UnsubstitutedConstraintSatisfactionCacheResult &R) -> bool {
    //       return Ctx.isAfterCheckpoint(static_cast<void
    //       *>(R.SubstExpr.get()),
    //                                    SlabCP)
    //     });
    assert(CP.UnsubstitutedConstraintSatisfactionCacheSize ==
           S.UnsubstitutedConstraintSatisfactionCache.size());
  }

  if (CP.SubsumptionCacheSize != S.SubsumptionCache.size()) {
    llvm::dbgs() << "CP.SubsumptionCacheSize != "
                    "S.SubsumptionCache.size()\n";
    // eraseDenseMapIf(
    //     S.SubsumptionCache,
    //     [&](llvm::DenseMap<std::pair<const NamedDecl *, const NamedDecl *>,
    //                        bool>::key_value &KV) -> bool {
    //       return Ctx.isAfterCheckpoint(static_cast<void *>(KV.first.first),
    //                                    SlabCP) ||
    //              Ctx.isAfterCheckpoint(static_cast<void *>(KV.first.second),
    //                                    SlabCP);
    //     });
    assert(CP.SubsumptionCacheSize == S.SubsumptionCache.size());
  }

  if (CP.NormalizationCacheSize != S.NormalizationCache.size()) {
    llvm::dbgs() << "CP.NormalizationCacheSize != "
                    "S.NormalizationCache.size()\n";
    //   llvm::DenseMap<ConstrainedDeclOrNestedRequirement, NormalizedConstraint
    //   *> NormalizationCache;
    assert(CP.NormalizationCacheSize == S.NormalizationCache.size());
  }

  if (CP.SatisfactionCacheSize != S.SatisfactionCache.size()) {
    llvm::dbgs() << "CP.SatisfactionCacheSize != "
                    "S.SatisfactionCache.size()\n";
    //   llvm::ContextualFoldingSet<ConstraintSatisfaction, const ASTContext &>
    //   SatisfactionCache;
    assert(CP.SatisfactionCacheSize == S.SatisfactionCache.size());
  }

  if (CP.SatisfactionStackSize != S.SatisfactionStack.size()) {
    llvm::dbgs() << "CP.SatisfactionStackSize != "
                    "S.SatisfactionStack.size()\n";
    //         llvm::SmallVector<SatisfactionStackEntryTy, 10>
    //         SatisfactionStack;
    assert(CP.SatisfactionStackSize == S.SatisfactionStack.size());
  }

  // if (CP.NullabilityMapSize != S.NullabilityMap.size()) {
  //   llvm::dbgs() << "CP.NullabilityMapSize != "
  //                   "S.NullabilityMap.size()\n";
  //   //  FileNullabilityMap NullabilityMap;;
  //   assert(CP.NullabilityMapSize == S.NullabilityMap.size());
  // }

  if (CP.DeclsWithEffectsToVerifySize != S.DeclsWithEffectsToVerify.size()) {
    llvm::dbgs() << "CP.DeclsWithEffectsToVerifySize != "
                    "S.DeclsWithEffectsToVerify.size()\n";
    //  SmallVector<const Decl *> DeclsWithEffectsToVerify;;
    assert(CP.DeclsWithEffectsToVerifySize ==
           S.DeclsWithEffectsToVerify.size());
  }

  if (CP.SpecialMemberCacheSize != S.SpecialMemberCache.size()) {
    llvm::dbgs() << "CP.SpecialMemberCacheSize != "
                    "S.SpecialMemberCache.size()\n";
    // eraseFoldingSetIf(S.SpecialMemberCache,
    //                   [&](Sema::SpecialMemberOverloadResultEntry &Node) ->
    //                   bool {
    //                     return S.BumpAlloc.isAfterCheckpoint(
    //                         static_cast<void *>(&Node), CP.SemaBumpSlabCP);
    //                   });
    assert(CP.SpecialMemberCacheSize == S.SpecialMemberCache.size());
  }

  std::vector<NamedDecl *> ToRemoved;
  for (auto *TmpD : S.getCurScope()->decls()) {
    assert(TmpD && "This decl didn't get pushed??");

    assert(isa<NamedDecl>(TmpD) && "Decl isn't NamedDecl?");
    NamedDecl *D = cast<NamedDecl>(TmpD);

    if (!D->getDeclName()) continue;

    if (Ctx.getAllocator().isAfterCheckpoint(static_cast<void *>(D), SlabCP)) {
      if (D->getDeclName().getFETokenInfo())
        S.IdResolver.RemoveDecl(D);
      ToRemoved.push_back(D);
    }
  }

  for (auto ND : ToRemoved)
    S.getCurScope()->RemoveDecl(ND);

  S.BumpAlloc.restoreToCheckPoint(CP.SemaBumpSlabCP);
}

} // namespace clang
