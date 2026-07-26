//===--- ASTContextStateStash.cpp - ASTContext persistent state stash/restore
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
#include "clang/AST/DeclContextInternals.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/DeclTemplate.h"
#include "clang/AST/Expr.h"
#include "clang/AST/ExprCXX.h"
#include "clang/AST/Type.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/FoldingSet.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/SmallVector.h"

namespace clang {

/// Erases nodes from a FoldingSet based on a predicate.
template <typename EntryType, typename PredT>
static void eraseFoldingSetIf(llvm::FoldingSet<EntryType> &FS, PredT &&Pred) {
  SmallVector<EntryType *, 16> ToRemove;

  for (auto &N : FS)
    if (Pred(N))
      ToRemove.push_back(&N);

  for (auto *N : ToRemove)
    FS.RemoveNode(N);
}

template <typename EntryType, typename PredT>
static void
eraseFoldingSetIf(llvm::ContextualFoldingSet<EntryType, ASTContext &> &FS,
                  PredT &&Pred) {
  SmallVector<EntryType *, 16> ToRemove;

  for (auto &N : FS)
    if (Pred(N))
      ToRemove.push_back(&N);

  for (auto *N : ToRemove)
    FS.RemoveNode(N);
}

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

/// Erase from DenseMap based on predicate
template <typename ValueT, typename PredT>
static void eraseDenseSetIf(llvm::DenseSet<ValueT> &Set, PredT &&Pred) {
  SmallVector<ValueT, 16> ToRemove;

  for (auto &V : Set)
    if (Pred(V))
      ToRemove.push_back(V);

  for (auto &Key : ToRemove)
    Set.erase(Key);
}

void ASTContextStateStash::stash(StashCheckPoint &CP) {
  CP.TypesSize = Ctx.Types.size();
  CP.ExtQualNodesSize = Ctx.ExtQualNodes.size();
  CP.ComplexTypesSize = Ctx.ComplexTypes.size();
  CP.PointerTypesSize = Ctx.PointerTypes.size();
  CP.AdjustedTypesSize = Ctx.AdjustedTypes.size();
  CP.BlockPointerTypesSize = Ctx.BlockPointerTypes.size();
  CP.LValueReferenceTypesSize = Ctx.LValueReferenceTypes.size();
  CP.RValueReferenceTypesSize = Ctx.RValueReferenceTypes.size();
  CP.MemberPointerTypesSize = Ctx.MemberPointerTypes.size();

  CP.ConstantArrayTypesSize = Ctx.ConstantArrayTypes.size();
  CP.IncompleteArrayTypesSize = Ctx.IncompleteArrayTypes.size();
  CP.VariableArrayTypesSize = Ctx.VariableArrayTypes.size();

  CP.DependentSizedArrayTypesSize = Ctx.DependentSizedArrayTypes.size();
  CP.DependentSizedExtVectorTypesSize = Ctx.DependentSizedExtVectorTypes.size();
  CP.DependentAddressSpaceTypesSize = Ctx.DependentAddressSpaceTypes.size();
  CP.VectorTypesSize = Ctx.VectorTypes.size();
  CP.DependentVectorTypesSize = Ctx.DependentVectorTypes.size();
  CP.MatrixTypesSize = Ctx.MatrixTypes.size();
  CP.DependentSizedMatrixTypesSize = Ctx.DependentSizedMatrixTypes.size();
  CP.FunctionNoProtoTypesSize = Ctx.FunctionNoProtoTypes.size();
  CP.FunctionProtoTypesSize = Ctx.FunctionProtoTypes.size();
  CP.DependentTypeOfExprTypesSize = Ctx.DependentTypeOfExprTypes.size();
  CP.DependentDecltypeTypesSize = Ctx.DependentDecltypeTypes.size();

  CP.DependentPackIndexingTypesSize = Ctx.DependentPackIndexingTypes.size();

  CP.TemplateTypeParmTypesSize = Ctx.TemplateTypeParmTypes.size();
  CP.ObjCTypeParamTypesSize = Ctx.ObjCTypeParamTypes.size();
  CP.SubstTemplateTypeParmTypesSize = Ctx.SubstTemplateTypeParmTypes.size();
  CP.SubstTemplateTypeParmPackTypesSize =
      Ctx.SubstTemplateTypeParmPackTypes.size();
  CP.SubstBuiltinTemplatePackTypesSize =
      Ctx.SubstBuiltinTemplatePackTypes.size();

  CP.TemplateSpecializationTypesSize = Ctx.TemplateSpecializationTypes.size();
  CP.ParenTypesSize = Ctx.ParenTypes.size();
  CP.TagTypesSize = Ctx.TagTypes.size();
  CP.UnresolvedUsingTypesSize = Ctx.UnresolvedUsingTypes.size();
  CP.UsingTypesSize = Ctx.UsingTypes.size();
  CP.TypedefTypesSize = Ctx.TypedefTypes.size();
  CP.DependentNameTypesSize = Ctx.DependentNameTypes.size();
  CP.PackExpansionTypesSize = Ctx.PackExpansionTypes.size();
  CP.ObjCObjectTypesSize = Ctx.ObjCObjectTypes.size();
  CP.ObjCObjectPointerTypesSize = Ctx.ObjCObjectPointerTypes.size();
  CP.UnaryTransformTypesSize = Ctx.UnaryTransformTypes.size();

  CP.AutoTypesSize = Ctx.AutoTypes.size();
  CP.DeducedTemplateSpecializationTypesSize =
      Ctx.DeducedTemplateSpecializationTypes.size();
  CP.AtomicTypesSize = Ctx.AtomicTypes.size();
  CP.AttributedTypesSize = Ctx.AttributedTypes.size();
  CP.PipeTypesSize = Ctx.PipeTypes.size();
  CP.BitIntTypesSize = Ctx.BitIntTypes.size();
  CP.DependentBitIntTypesSize = Ctx.DependentBitIntTypes.size();
  CP.BTFTagAttributedTypesSize = Ctx.BTFTagAttributedTypes.size();
  CP.HLSLAttributedResourceTypesSize = Ctx.HLSLAttributedResourceTypes.size();
  CP.HLSLInlineSpirvTypesSize = Ctx.HLSLInlineSpirvTypes.size();

  CP.CountAttributedTypesSize = Ctx.CountAttributedTypes.size();

  CP.QualifiedTemplateNamesSize = Ctx.QualifiedTemplateNames.size();
  CP.DependentTemplateNamesSize = Ctx.DependentTemplateNames.size();
  CP.SubstTemplateTemplateParmsSize = Ctx.SubstTemplateTemplateParms.size();
  CP.SubstTemplateTemplateParmPacksSize =
      Ctx.SubstTemplateTemplateParmPacks.size();
  CP.DeducedTemplatesSize = Ctx.DeducedTemplates.size();

  CP.ArrayParameterTypesSize = Ctx.ArrayParameterTypes.size();

  CP.PredefinedSugarTypesSize = Ctx.PredefinedSugarTypes.size();

  CP.NamespaceAndPrefixStoragesSize = Ctx.NamespaceAndPrefixStorages.size();

  CP.ASTRecordLayoutsSize = Ctx.ASTRecordLayouts.size();

  CP.MemoizedTypeInfoSize = Ctx.MemoizedTypeInfo.size();

  CP.MemoizedUnadjustedAlignSize = Ctx.MemoizedUnadjustedAlign.size();

  CP.KeyFunctionsSize = Ctx.KeyFunctions.size();

  CP.BlockVarCopyInitsSize = Ctx.BlockVarCopyInits.size();

  CP.MSGuidDeclsSize = Ctx.MSGuidDecls.size();

  CP.UnnamedGlobalConstantDeclsSize = Ctx.UnnamedGlobalConstantDecls.size();

  CP.TemplateParamObjectDeclsSize = Ctx.TemplateParamObjectDecls.size();

  CP.StringLiteralCacheSize = Ctx.StringLiteralCache.size();

  CP.DestroyingOperatorDeletesSize = Ctx.DestroyingOperatorDeletes.size();
  CP.TypeAwareOperatorNewAndDeletesSize =
      Ctx.TypeAwareOperatorNewAndDeletes.size();

  CP.OperatorDeletesForVirtualDtorSize =
      Ctx.OperatorDeletesForVirtualDtor.size();

  CP.GlobalOperatorDeletesForVirtualDtorSize =
      Ctx.GlobalOperatorDeletesForVirtualDtor.size();

  CP.ArrayOperatorDeletesForVirtualDtorSize =
      Ctx.ArrayOperatorDeletesForVirtualDtor.size();
  CP.GlobalArrayOperatorDeletesForVirtualDtorSize =
      Ctx.GlobalArrayOperatorDeletesForVirtualDtor.size();

  CP.RequireVectorDeletingDtorSize = Ctx.RequireVectorDeletingDtor.size();

  CP.MergedDeclsSize = Ctx.MergedDecls.size();
  CP.MergedDefModulesSize = Ctx.MergedDefModules.size();

  CP.ModuleInitializersSize = Ctx.ModuleInitializers.size();
  CP.PrimaryModuleNameMapSize = Ctx.PrimaryModuleNameMap.size();
  CP.SameModuleLookupSetSize = Ctx.SameModuleLookupSet.size();

  CP.ScalableVecTyMapSize = Ctx.ScalableVecTyMap.size();
  CP.LambdaCastPathsSize = Ctx.LambdaCastPaths.size();
  CP.DeclRawCommentsSize = Ctx.DeclRawComments.size();
  CP.RedeclChainCommentsSize = Ctx.RedeclChainComments.size();
  CP.CommentlessRedeclChainsSize = Ctx.CommentlessRedeclChains.size();
  CP.ParsedCommentsSize = Ctx.ParsedComments.size();
  CP.RelocatableClassesSize = Ctx.RelocatableClasses.size();
  CP.ParamIndicesSize = Ctx.ParamIndices.size();
  CP.MangleNumbersSize = Ctx.MangleNumbers.size();
  CP.StaticLocalNumbersSize = Ctx.StaticLocalNumbers.size();
  CP.TemplateOrInstantiationSize = Ctx.TemplateOrInstantiation.size();

  CP.InstantiatedFromUsingDeclSize = Ctx.InstantiatedFromUsingDecl.size();
  CP.InstantiatedFromUsingEnumDeclSize =
      Ctx.InstantiatedFromUsingEnumDecl.size();

  CP.InstantiatedFromUsingShadowDeclSize =
      Ctx.InstantiatedFromUsingShadowDecl.size();

  CP.InstantiatedFromUnnamedFieldDeclSize =
      Ctx.InstantiatedFromUnnamedFieldDecl.size();
  CP.OverriddenMethodsSize = Ctx.OverriddenMethods.size();
  CP.MangleNumberingContextsSize = Ctx.MangleNumberingContexts.size();
  CP.ExtraMangleNumberingContextsSize = Ctx.ExtraMangleNumberingContexts.size();
  CP.TraversalScopeSize = Ctx.TraversalScope.size();
  CP.LastSDM = Ctx.LastSDM;
}

void ASTContextStateStash::restore(StashCheckPoint &CP,
                                   llvm::SlabCheckPoint SlabCP) {

  if (Ctx.TemplateTypeParmTypes.size() != CP.TemplateTypeParmTypesSize) {
    llvm::dbgs() << "Ctx.TemplateTypeParmTypes.size() != "
                    "CP.TemplateTypeParmTypesSize\n";
    eraseFoldingSetIf(Ctx.TemplateTypeParmTypes,
                      [&](TemplateTypeParmType &Node) -> bool {
                        return Ctx.getAllocator().isAfterCheckpoint(
                            static_cast<void *>(&Node), SlabCP);
                      });
    assert(Ctx.TemplateTypeParmTypes.size() == CP.TemplateTypeParmTypesSize);
  }

  if (Ctx.TagTypes.size() != CP.TagTypesSize) {
    llvm::dbgs() << "Ctx.TagTypes.size() != CP.TagTypesSize\n";
    eraseFoldingSetIf(Ctx.TagTypes,
                      [&](TagTypeFoldingSetPlaceholder &Node) -> bool {
                        return Ctx.getAllocator().isAfterCheckpoint(
                            static_cast<void *>(&Node), SlabCP);
                      });

    assert(Ctx.TagTypes.size() == CP.TagTypesSize);
  }

  if (Ctx.ExtQualNodes.size() != CP.ExtQualNodesSize) {
    llvm::dbgs() << "Ctx.ExtQualNodes.size() != CP.ExtQualNodesSize\n";
    eraseFoldingSetIf(Ctx.ExtQualNodes, [&](ExtQuals &Node) -> bool {
      return Ctx.getAllocator().isAfterCheckpoint(static_cast<void *>(&Node),
                                                  SlabCP);
    });
    assert(Ctx.ExtQualNodes.size() == CP.ExtQualNodesSize);
  }

  if (Ctx.ComplexTypes.size() != CP.ComplexTypesSize) {
    llvm::dbgs() << "Ctx.ComplexTypes.size() != CP.ComplexTypesSize\n";
    eraseFoldingSetIf(Ctx.ComplexTypes, [&](ComplexType &Node) -> bool {
      return Ctx.getAllocator().isAfterCheckpoint(static_cast<void *>(&Node),
                                                  SlabCP);
    });
    assert(Ctx.ComplexTypes.size() == CP.ComplexTypesSize);
  }

  if (Ctx.PointerTypes.size() != CP.PointerTypesSize) {
    llvm::dbgs() << "Ctx.PointerTypes.size() != CP.PointerTypesSize\n";
    eraseFoldingSetIf(Ctx.PointerTypes, [&](PointerType &Node) -> bool {
      return Ctx.getAllocator().isAfterCheckpoint(static_cast<void *>(&Node),
                                                  SlabCP);
    });
    assert(Ctx.PointerTypes.size() == CP.PointerTypesSize);
  }

  if (Ctx.AdjustedTypes.size() != CP.AdjustedTypesSize) {
    llvm::dbgs() << "Ctx.AdjustedTypes.size() != CP.AdjustedTypesSize\n";
    eraseFoldingSetIf(Ctx.AdjustedTypes, [&](AdjustedType &Node) -> bool {
      return Ctx.getAllocator().isAfterCheckpoint(static_cast<void *>(&Node),
                                                  SlabCP);
    });
    assert(Ctx.AdjustedTypes.size() == CP.AdjustedTypesSize);
  }

  if (Ctx.BlockPointerTypes.size() != CP.BlockPointerTypesSize) {
    llvm::dbgs()
        << "Ctx.BlockPointerTypes.size() != CP.BlockPointerTypesSize\n";
    eraseFoldingSetIf(Ctx.BlockPointerTypes,
                      [&](BlockPointerType &Node) -> bool {
                        return Ctx.getAllocator().isAfterCheckpoint(
                            static_cast<void *>(&Node), SlabCP);
                      });
    assert(Ctx.BlockPointerTypes.size() == CP.BlockPointerTypesSize);
  }

  if (Ctx.LValueReferenceTypes.size() != CP.LValueReferenceTypesSize) {
    llvm::dbgs()
        << "Ctx.LValueReferenceTypes.size() != CP.LValueReferenceTypesSize\n";
    eraseFoldingSetIf(Ctx.LValueReferenceTypes,
                      [&](LValueReferenceType &Node) -> bool {
                        return Ctx.getAllocator().isAfterCheckpoint(
                            static_cast<void *>(&Node), SlabCP);
                      });
    assert(Ctx.LValueReferenceTypes.size() == CP.LValueReferenceTypesSize);
  }

  if (Ctx.RValueReferenceTypes.size() != CP.RValueReferenceTypesSize) {
    llvm::dbgs()
        << "Ctx.RValueReferenceTypes.size() != CP.RValueReferenceTypesSize\n";
    eraseFoldingSetIf(Ctx.RValueReferenceTypes,
                      [&](RValueReferenceType &Node) -> bool {
                        return Ctx.getAllocator().isAfterCheckpoint(
                            static_cast<void *>(&Node), SlabCP);
                      });
    assert(Ctx.RValueReferenceTypes.size() == CP.RValueReferenceTypesSize);
  }

  if (Ctx.MemberPointerTypes.size() != CP.MemberPointerTypesSize) {
    llvm::dbgs()
        << "Ctx.MemberPointerTypes.size() != CP.MemberPointerTypesSize\n";
    eraseFoldingSetIf(Ctx.MemberPointerTypes,
                      [&](MemberPointerType &Node) -> bool {
                        return Ctx.getAllocator().isAfterCheckpoint(
                            static_cast<void *>(&Node), SlabCP);
                      });
    assert(Ctx.MemberPointerTypes.size() == CP.MemberPointerTypesSize);
  }

  if (Ctx.ConstantArrayTypes.size() != CP.ConstantArrayTypesSize) {
    llvm::dbgs()
        << "Ctx.ConstantArrayTypes.size() != CP.ConstantArrayTypesSize\n";
    eraseFoldingSetIf(Ctx.ConstantArrayTypes,
                      [&](ConstantArrayType &Node) -> bool {
                        return Ctx.getAllocator().isAfterCheckpoint(
                            static_cast<void *>(&Node), SlabCP);
                      });
    assert(Ctx.ConstantArrayTypes.size() == CP.ConstantArrayTypesSize);
  }

  if (Ctx.IncompleteArrayTypes.size() != CP.IncompleteArrayTypesSize) {
    llvm::dbgs()
        << "Ctx.IncompleteArrayTypes.size() != CP.IncompleteArrayTypesSize\n";
    eraseFoldingSetIf(Ctx.IncompleteArrayTypes,
                      [&](IncompleteArrayType &Node) -> bool {
                        return Ctx.getAllocator().isAfterCheckpoint(
                            static_cast<void *>(&Node), SlabCP);
                      });
    assert(Ctx.IncompleteArrayTypes.size() == CP.IncompleteArrayTypesSize);
  }

  // mutable std::vector<VariableArrayType*> VariableArrayTypes;
  if (Ctx.VariableArrayTypes.size() != CP.VariableArrayTypesSize) {
    llvm::dbgs()
        << "Ctx.VariableArrayTypes.size() != CP.VariableArrayTypesSize\n";
    Ctx.VariableArrayTypes.resize(CP.VariableArrayTypesSize);
    assert(Ctx.VariableArrayTypes.size() == CP.VariableArrayTypesSize);
  }

  if (Ctx.DependentSizedArrayTypes.size() != CP.DependentSizedArrayTypesSize) {
    llvm::dbgs() << "Ctx.DependentSizedArrayTypes.size() != "
                    "CP.DependentSizedArrayTypesSize\n";
    eraseFoldingSetIf(Ctx.DependentSizedArrayTypes,
                      [&](DependentSizedArrayType &Node) -> bool {
                        return Ctx.getAllocator().isAfterCheckpoint(
                            static_cast<void *>(&Node), SlabCP);
                      });
    assert(Ctx.DependentSizedArrayTypes.size() ==
           CP.DependentSizedArrayTypesSize);
  }

  if (Ctx.DependentSizedExtVectorTypes.size() !=
      CP.DependentSizedExtVectorTypesSize) {
    llvm::dbgs() << "Ctx.DependentSizedExtVectorTypes.size() != "
                    "CP.DependentSizedExtVectorTypesSize\n";
    eraseFoldingSetIf(Ctx.DependentSizedExtVectorTypes,
                      [&](DependentSizedExtVectorType &Node) -> bool {
                        return Ctx.getAllocator().isAfterCheckpoint(
                            static_cast<void *>(&Node), SlabCP);
                      });
    assert(Ctx.DependentSizedExtVectorTypes.size() ==
           CP.DependentSizedExtVectorTypesSize);
  }

  if (Ctx.DependentAddressSpaceTypes.size() !=
      CP.DependentAddressSpaceTypesSize) {
    llvm::dbgs() << "Ctx.DependentAddressSpaceTypes.size() != "
                    "CP.DependentAddressSpaceTypesSize\n";

    eraseFoldingSetIf(Ctx.DependentAddressSpaceTypes,
                      [&](DependentAddressSpaceType &Node) -> bool {
                        return Ctx.getAllocator().isAfterCheckpoint(
                            static_cast<void *>(&Node), SlabCP);
                      });
    assert(Ctx.DependentAddressSpaceTypes.size() ==
           CP.DependentAddressSpaceTypesSize);
  }

  if (Ctx.VectorTypes.size() != CP.VectorTypesSize) {
    llvm::dbgs() << "Ctx.VectorTypes.size() != CP.VectorTypesSize\n";
    eraseFoldingSetIf(Ctx.VectorTypes, [&](VectorType &Node) -> bool {
      return Ctx.getAllocator().isAfterCheckpoint(static_cast<void *>(&Node),
                                                  SlabCP);
    });
    assert(Ctx.VectorTypes.size() == CP.VectorTypesSize);
  }

  if (Ctx.DependentVectorTypes.size() != CP.DependentVectorTypesSize) {
    llvm::dbgs()
        << "Ctx.DependentVectorTypes.size() != CP.DependentVectorTypesSize\n";
    eraseFoldingSetIf(Ctx.DependentVectorTypes,
                      [&](DependentVectorType &Node) -> bool {
                        return Ctx.getAllocator().isAfterCheckpoint(
                            static_cast<void *>(&Node), SlabCP);
                      });
    assert(Ctx.DependentVectorTypes.size() == CP.DependentVectorTypesSize);
  }

  if (Ctx.MatrixTypes.size() != CP.MatrixTypesSize) {
    llvm::dbgs() << "Ctx.MatrixTypes.size() != CP.MatrixTypesSize\n";
    eraseFoldingSetIf(Ctx.MatrixTypes, [&](ConstantMatrixType &Node) -> bool {
      return Ctx.getAllocator().isAfterCheckpoint(static_cast<void *>(&Node),
                                                  SlabCP);
    });
    assert(Ctx.MatrixTypes.size() == CP.MatrixTypesSize);
  }

  if (Ctx.DependentSizedMatrixTypes.size() !=
      CP.DependentSizedMatrixTypesSize) {
    llvm::dbgs() << "Ctx.DependentSizedMatrixTypes.size() != "
                    "CP.DependentSizedMatrixTypesSize\n";
    eraseFoldingSetIf(Ctx.DependentSizedMatrixTypes,
                      [&](DependentSizedMatrixType &Node) -> bool {
                        return Ctx.getAllocator().isAfterCheckpoint(
                            static_cast<void *>(&Node), SlabCP);
                      });
    assert(Ctx.DependentSizedMatrixTypes.size() ==
           CP.DependentSizedMatrixTypesSize);
  }

  if (Ctx.FunctionNoProtoTypes.size() != CP.FunctionNoProtoTypesSize) {
    llvm::dbgs()
        << "Ctx.FunctionNoProtoTypes.size() != CP.FunctionNoProtoTypesSize\n";
    eraseFoldingSetIf(Ctx.FunctionNoProtoTypes,
                      [&](FunctionNoProtoType &Node) -> bool {
                        return Ctx.getAllocator().isAfterCheckpoint(
                            static_cast<void *>(&Node), SlabCP);
                      });
    assert(Ctx.FunctionNoProtoTypes.size() == CP.FunctionNoProtoTypesSize);
  }

  if (Ctx.FunctionProtoTypes.size() != CP.FunctionProtoTypesSize) {
    llvm::dbgs()
        << "Ctx.FunctionProtoTypes.size() != CP.FunctionProtoTypesSize\n";
    eraseFoldingSetIf(Ctx.FunctionProtoTypes,
                      [&](FunctionProtoType &Node) -> bool {
                        return Ctx.getAllocator().isAfterCheckpoint(
                            static_cast<void *>(&Node), SlabCP);
                      });
    assert(Ctx.FunctionProtoTypes.size() == CP.FunctionProtoTypesSize);
  }

  if (Ctx.DependentTypeOfExprTypes.size() != CP.DependentTypeOfExprTypesSize) {
    llvm::dbgs() << "Ctx.DependentTypeOfExprTypes.size() != "
                    "CP.DependentTypeOfExprTypesSize\n";
    eraseFoldingSetIf(Ctx.DependentTypeOfExprTypes,
                      [&](DependentTypeOfExprType &Node) -> bool {
                        return Ctx.getAllocator().isAfterCheckpoint(
                            static_cast<void *>(&Node), SlabCP);
                      });
    assert(Ctx.DependentTypeOfExprTypes.size() ==
           CP.DependentTypeOfExprTypesSize);
  }

  if (Ctx.DependentDecltypeTypes.size() != CP.DependentDecltypeTypesSize) {
    llvm::dbgs() << "Ctx.DependentDecltypeTypes.size() != "
                    "CP.DependentDecltypeTypesSize\n";
    eraseFoldingSetIf(Ctx.DependentDecltypeTypes,
                      [&](DependentDecltypeType &Node) -> bool {
                        return Ctx.getAllocator().isAfterCheckpoint(
                            static_cast<void *>(&Node), SlabCP);
                      });
    assert(Ctx.DependentDecltypeTypes.size() == CP.DependentDecltypeTypesSize);
  }

  if (Ctx.DependentPackIndexingTypes.size() !=
      CP.DependentPackIndexingTypesSize) {
    llvm::dbgs() << "Ctx.DependentPackIndexingTypes.size() != "
                    "CP.DependentPackIndexingTypesSize\n";
    eraseFoldingSetIf(Ctx.DependentPackIndexingTypes,
                      [&](PackIndexingType &Node) -> bool {
                        return Ctx.getAllocator().isAfterCheckpoint(
                            static_cast<void *>(&Node), SlabCP);
                      });
    assert(Ctx.DependentPackIndexingTypes.size() ==
           CP.DependentPackIndexingTypesSize);
  }

  if (Ctx.TemplateTypeParmTypes.size() != CP.TemplateTypeParmTypesSize) {
    llvm::dbgs() << "Ctx.TemplateTypeParmTypes.size() != "
                    "CP.TemplateTypeParmTypesSize\n";
    eraseFoldingSetIf(Ctx.TemplateTypeParmTypes,
                      [&](TemplateTypeParmType &Node) -> bool {
                        return Ctx.getAllocator().isAfterCheckpoint(
                            static_cast<void *>(&Node), SlabCP);
                      });
    assert(Ctx.TemplateTypeParmTypes.size() == CP.TemplateTypeParmTypesSize);
  }

  // mutable llvm::FoldingSet<ObjCTypeParamType> ObjCTypeParamTypes;

  if (Ctx.SubstTemplateTypeParmTypes.size() !=
      CP.SubstTemplateTypeParmTypesSize) {
    llvm::dbgs() << "Ctx.SubstTemplateTypeParmTypes.size() != "
                    "CP.SubstTemplateTypeParmTypesSize\n";
    eraseFoldingSetIf(Ctx.SubstTemplateTypeParmTypes,
                      [&](SubstTemplateTypeParmType &Node) -> bool {
                        return Ctx.getAllocator().isAfterCheckpoint(
                            static_cast<void *>(&Node), SlabCP);
                      });
    assert(Ctx.SubstTemplateTypeParmTypes.size() ==
           CP.SubstTemplateTypeParmTypesSize);
  }

  if (Ctx.SubstTemplateTypeParmPackTypes.size() !=
      CP.SubstTemplateTypeParmPackTypesSize) {
    llvm::dbgs() << "Ctx.SubstTemplateTypeParmPackTypes.size() != "
                    "CP.SubstTemplateTypeParmPackTypesSize\n";
    eraseFoldingSetIf(Ctx.SubstTemplateTypeParmPackTypes,
                      [&](SubstTemplateTypeParmPackType &Node) -> bool {
                        return Ctx.getAllocator().isAfterCheckpoint(
                            static_cast<void *>(&Node), SlabCP);
                      });
    assert(Ctx.SubstTemplateTypeParmPackTypes.size() ==
           CP.SubstTemplateTypeParmPackTypesSize);
  }

  if (Ctx.SubstBuiltinTemplatePackTypes.size() !=
      CP.SubstBuiltinTemplatePackTypesSize) {
    llvm::dbgs() << "Ctx.SubstBuiltinTemplatePackTypes.size() != "
                    "CP.SubstBuiltinTemplatePackTypesSize\n";
    eraseFoldingSetIf(Ctx.SubstBuiltinTemplatePackTypes,
                      [&](SubstBuiltinTemplatePackType &Node) -> bool {
                        return Ctx.getAllocator().isAfterCheckpoint(
                            static_cast<void *>(&Node), SlabCP);
                      });
    assert(Ctx.SubstBuiltinTemplatePackTypes.size() ==
           CP.SubstBuiltinTemplatePackTypesSize);
  }

  if (Ctx.TemplateSpecializationTypes.size() !=
      CP.TemplateSpecializationTypesSize) {
    llvm::dbgs() << "Ctx.TemplateSpecializationTypes.size() != "
                    "CP.TemplateSpecializationTypesSize\n";
    eraseFoldingSetIf(Ctx.TemplateSpecializationTypes,
                      [&](TemplateSpecializationType &Node) -> bool {
                        return Ctx.getAllocator().isAfterCheckpoint(
                            static_cast<void *>(&Node), SlabCP);
                      });
    assert(Ctx.TemplateSpecializationTypes.size() ==
           CP.TemplateSpecializationTypesSize);
  }

  if (Ctx.ParenTypes.size() != CP.ParenTypesSize) {
    llvm::dbgs() << "Ctx.ParenTypes.size() != CP.ParenTypesSize\n";
    eraseFoldingSetIf(Ctx.ParenTypes, [&](ParenType &Node) -> bool {
      return Ctx.getAllocator().isAfterCheckpoint(static_cast<void *>(&Node),
                                                  SlabCP);
    });
    assert(Ctx.ParenTypes.size() == CP.ParenTypesSize);
  }

  if (Ctx.TagTypes.size() != CP.TagTypesSize) {
    llvm::dbgs() << "Ctx.TagTypes.size() != CP.TagTypesSize\n";
    eraseFoldingSetIf(Ctx.TagTypes,
                      [&](TagTypeFoldingSetPlaceholder &Node) -> bool {
                        return Ctx.getAllocator().isAfterCheckpoint(
                            static_cast<void *>(&Node), SlabCP);
                      });
    assert(Ctx.TagTypes.size() == CP.TagTypesSize);
  }

  if (Ctx.UnresolvedUsingTypes.size() != CP.UnresolvedUsingTypesSize) {
    llvm::dbgs()
        << "Ctx.UnresolvedUsingTypes.size() != CP.UnresolvedUsingTypesSize\n";
    eraseFoldingSetIf(
        Ctx.UnresolvedUsingTypes,
        [&](FoldingSetPlaceholder<UnresolvedUsingType> &Node) -> bool {
          return Ctx.getAllocator().isAfterCheckpoint(
              static_cast<void *>(&Node), SlabCP);
        });
    assert(Ctx.UnresolvedUsingTypes.size() == CP.UnresolvedUsingTypesSize);
  }

  if (Ctx.UsingTypes.size() != CP.UsingTypesSize) {
    llvm::dbgs() << "Ctx.UsingTypes.size() != CP.UsingTypesSize\n";
    eraseFoldingSetIf(Ctx.UsingTypes, [&](UsingType &Node) -> bool {
      return Ctx.getAllocator().isAfterCheckpoint(static_cast<void *>(&Node),
                                                  SlabCP);
    });
    assert(Ctx.UsingTypes.size() == CP.UsingTypesSize);
  }

  if (Ctx.TypedefTypes.size() != CP.TypedefTypesSize) {
    llvm::dbgs() << "Ctx.TypedefTypes.size() != CP.TypedefTypesSize\n";
    eraseFoldingSetIf(Ctx.TypedefTypes,
                      [&](FoldingSetPlaceholder<TypedefType> &Node) -> bool {
                        return Ctx.getAllocator().isAfterCheckpoint(
                            static_cast<void *>(&Node), SlabCP);
                      });
    assert(Ctx.TypedefTypes.size() == CP.TypedefTypesSize);
  }

  if (Ctx.DependentNameTypes.size() != CP.DependentNameTypesSize) {
    llvm::dbgs()
        << "Ctx.DependentNameTypes.size() != CP.DependentNameTypesSize\n";
    eraseFoldingSetIf(Ctx.DependentNameTypes,
                      [&](DependentNameType &Node) -> bool {
                        return Ctx.getAllocator().isAfterCheckpoint(
                            static_cast<void *>(&Node), SlabCP);
                      });
    assert(Ctx.DependentNameTypes.size() == CP.DependentNameTypesSize);
  }

  if (Ctx.PackExpansionTypes.size() != CP.PackExpansionTypesSize) {
    llvm::dbgs()
        << "Ctx.PackExpansionTypes.size() != CP.PackExpansionTypesSize\n";
    eraseFoldingSetIf(Ctx.PackExpansionTypes,
                      [&](PackExpansionType &Node) -> bool {
                        return Ctx.getAllocator().isAfterCheckpoint(
                            static_cast<void *>(&Node), SlabCP);
                      });
    assert(Ctx.PackExpansionTypes.size() == CP.PackExpansionTypesSize);
  }

  // mutable llvm::FoldingSet<ObjCObjectTypeImpl> ObjCObjectTypes;
  // mutable llvm::FoldingSet<ObjCObjectPointerType> ObjCObjectPointerTypes;

  if (Ctx.UnaryTransformTypes.size() != CP.UnaryTransformTypesSize) {
    llvm::dbgs()
        << "Ctx.UnaryTransformTypes.size() != CP.UnaryTransformTypesSize\n";
    eraseFoldingSetIf(Ctx.UnaryTransformTypes,
                      [&](UnaryTransformType &Node) -> bool {
                        return Ctx.getAllocator().isAfterCheckpoint(
                            static_cast<void *>(&Node), SlabCP);
                      });
    assert(Ctx.UnaryTransformTypes.size() == CP.UnaryTransformTypesSize);
  }

  if (Ctx.AutoTypes.size() != CP.AutoTypesSize) {
    llvm::dbgs() << "Ctx.AutoTypes.size() != CP.AutoTypesSize\n";
    // mutable llvm::DenseMap<llvm::FoldingSetNodeID, AutoType *> AutoTypes;
    eraseDenseMapIf(
        Ctx.AutoTypes,
        [&](llvm::detail::DenseMapPair<llvm::FoldingSetNodeID, AutoType *> &KV)
            -> bool {
          return Ctx.getAllocator().isAfterCheckpoint(
              static_cast<void *>(KV.getSecond()), SlabCP);
        });
    assert(Ctx.AutoTypes.size() == CP.AutoTypesSize);
  }

  if (Ctx.DeducedTemplateSpecializationTypes.size() !=
      CP.DeducedTemplateSpecializationTypesSize) {
    llvm::dbgs() << "Ctx.DeducedTemplateSpecializationTypes.size() != "
                    "CP.DeducedTemplateSpecializationTypesSize\n";
    eraseFoldingSetIf(Ctx.DeducedTemplateSpecializationTypes,
                      [&](DeducedTemplateSpecializationType &Node) -> bool {
                        return Ctx.getAllocator().isAfterCheckpoint(
                            static_cast<void *>(&Node), SlabCP);
                      });
    assert(Ctx.DeducedTemplateSpecializationTypes.size() ==
           CP.DeducedTemplateSpecializationTypesSize);
  }

  if (Ctx.AtomicTypes.size() != CP.AtomicTypesSize) {
    llvm::dbgs() << "Ctx.AtomicTypes.size() != CP.AtomicTypesSize\n";
    eraseFoldingSetIf(Ctx.AtomicTypes, [&](AtomicType &Node) -> bool {
      return Ctx.getAllocator().isAfterCheckpoint(static_cast<void *>(&Node),
                                                  SlabCP);
    });
    assert(Ctx.AtomicTypes.size() == CP.AtomicTypesSize);
  }

  if (Ctx.AttributedTypes.size() != CP.AttributedTypesSize) {
    llvm::dbgs() << "Ctx.AttributedTypes.size() != CP.AttributedTypesSize\n";
    eraseFoldingSetIf(Ctx.AttributedTypes, [&](AttributedType &Node) -> bool {
      return Ctx.getAllocator().isAfterCheckpoint(static_cast<void *>(&Node),
                                                  SlabCP);
    });
    assert(Ctx.AttributedTypes.size() == CP.AttributedTypesSize);
  }

  if (Ctx.PipeTypes.size() != CP.PipeTypesSize) {
    llvm::dbgs() << "Ctx.PipeTypes.size() != CP.PipeTypesSize\n";
    eraseFoldingSetIf(Ctx.PipeTypes, [&](PipeType &Node) -> bool {
      return Ctx.getAllocator().isAfterCheckpoint(static_cast<void *>(&Node),
                                                  SlabCP);
    });
    assert(Ctx.PipeTypes.size() == CP.PipeTypesSize);
  }

  if (Ctx.BitIntTypes.size() != CP.BitIntTypesSize) {
    llvm::dbgs() << "Ctx.BitIntTypes.size() != CP.BitIntTypesSize\n";
    eraseFoldingSetIf(Ctx.BitIntTypes, [&](BitIntType &Node) -> bool {
      return Ctx.getAllocator().isAfterCheckpoint(static_cast<void *>(&Node),
                                                  SlabCP);
    });
    assert(Ctx.BitIntTypes.size() == CP.BitIntTypesSize);
  }

  if (Ctx.DependentBitIntTypes.size() != CP.DependentBitIntTypesSize) {
    llvm::dbgs()
        << "Ctx.DependentBitIntTypes.size() != CP.DependentBitIntTypesSize\n";
    eraseFoldingSetIf(Ctx.DependentBitIntTypes,
                      [&](DependentBitIntType &Node) -> bool {
                        return Ctx.getAllocator().isAfterCheckpoint(
                            static_cast<void *>(&Node), SlabCP);
                      });
    assert(Ctx.DependentBitIntTypes.size() == CP.DependentBitIntTypesSize);
  }

  if (Ctx.BTFTagAttributedTypes.size() != CP.BTFTagAttributedTypesSize) {
    llvm::dbgs() << "Ctx.BTFTagAttributedTypes.size() != "
                    "CP.BTFTagAttributedTypesSize\n";
    eraseFoldingSetIf(Ctx.BTFTagAttributedTypes,
                      [&](BTFTagAttributedType &Node) -> bool {
                        return Ctx.getAllocator().isAfterCheckpoint(
                            static_cast<void *>(&Node), SlabCP);
                      });
    assert(Ctx.BTFTagAttributedTypes.size() == CP.BTFTagAttributedTypesSize);
  }

  if (Ctx.HLSLAttributedResourceTypes.size() !=
      CP.HLSLAttributedResourceTypesSize) {
    llvm::dbgs() << "Ctx.HLSLAttributedResourceTypes.size() != "
                    "CP.HLSLAttributedResourceTypesSize\n";
    eraseFoldingSetIf(Ctx.HLSLAttributedResourceTypes,
                      [&](HLSLAttributedResourceType &Node) -> bool {
                        return Ctx.getAllocator().isAfterCheckpoint(
                            static_cast<void *>(&Node), SlabCP);
                      });
    assert(Ctx.HLSLAttributedResourceTypes.size() ==
           CP.HLSLAttributedResourceTypesSize);
  }

  if (Ctx.HLSLInlineSpirvTypes.size() != CP.HLSLInlineSpirvTypesSize) {
    llvm::dbgs()
        << "Ctx.HLSLInlineSpirvTypes.size() != CP.HLSLInlineSpirvTypesSize\n";
    eraseFoldingSetIf(Ctx.HLSLInlineSpirvTypes,
                      [&](HLSLInlineSpirvType &Node) -> bool {
                        return Ctx.getAllocator().isAfterCheckpoint(
                            static_cast<void *>(&Node), SlabCP);
                      });
    assert(Ctx.HLSLInlineSpirvTypes.size() == CP.HLSLInlineSpirvTypesSize);
  }

  if (Ctx.CountAttributedTypes.size() != CP.CountAttributedTypesSize) {
    llvm::dbgs()
        << "Ctx.CountAttributedTypes.size() != CP.CountAttributedTypesSize\n";
    eraseFoldingSetIf(Ctx.CountAttributedTypes,
                      [&](CountAttributedType &Node) -> bool {
                        return Ctx.getAllocator().isAfterCheckpoint(
                            static_cast<void *>(&Node), SlabCP);
                      });
    assert(Ctx.CountAttributedTypes.size() == CP.CountAttributedTypesSize);
  }

  if (Ctx.QualifiedTemplateNames.size() != CP.QualifiedTemplateNamesSize) {
    llvm::dbgs() << "Ctx.QualifiedTemplateNames.size() != "
                    "CP.QualifiedTemplateNamesSize\n";
    eraseFoldingSetIf(Ctx.QualifiedTemplateNames,
                      [&](QualifiedTemplateName &Node) -> bool {
                        return Ctx.getAllocator().isAfterCheckpoint(
                            static_cast<void *>(&Node), SlabCP);
                      });
    assert(Ctx.QualifiedTemplateNames.size() == CP.QualifiedTemplateNamesSize);
  }

  if (Ctx.DependentTemplateNames.size() != CP.DependentTemplateNamesSize) {
    llvm::dbgs() << "Ctx.DependentTemplateNames.size() != "
                    "CP.DependentTemplateNamesSize\n";
    eraseFoldingSetIf(Ctx.DependentTemplateNames,
                      [&](DependentTemplateName &Node) -> bool {
                        return Ctx.getAllocator().isAfterCheckpoint(
                            static_cast<void *>(&Node), SlabCP);
                      });
    assert(Ctx.DependentTemplateNames.size() == CP.DependentTemplateNamesSize);
  }

  if (Ctx.SubstTemplateTemplateParms.size() !=
      CP.SubstTemplateTemplateParmsSize) {
    llvm::dbgs() << "Ctx.SubstTemplateTemplateParms.size() != "
                    "CP.SubstTemplateTemplateParmsSize\n";
    eraseFoldingSetIf(Ctx.SubstTemplateTemplateParms,
                      [&](SubstTemplateTemplateParmStorage &Node) -> bool {
                        return Ctx.getAllocator().isAfterCheckpoint(
                            static_cast<void *>(&Node), SlabCP);
                      });
    assert(Ctx.SubstTemplateTemplateParms.size() ==
           CP.SubstTemplateTemplateParmsSize);
  }

  if (Ctx.SubstTemplateTemplateParmPacks.size() !=
      CP.SubstTemplateTemplateParmPacksSize) {
    llvm::dbgs() << "Ctx.SubstTemplateTemplateParmPacks.size() != "
                    "CP.SubstTemplateTemplateParmPacksSize\n";
    eraseFoldingSetIf(Ctx.SubstTemplateTemplateParmPacks,
                      [&](SubstTemplateTemplateParmPackStorage &Node) -> bool {
                        return Ctx.getAllocator().isAfterCheckpoint(
                            static_cast<void *>(&Node), SlabCP);
                      });
    assert(Ctx.SubstTemplateTemplateParmPacks.size() ==
           CP.SubstTemplateTemplateParmPacksSize);
  }

  if (Ctx.DeducedTemplates.size() != CP.DeducedTemplatesSize) {
    llvm::dbgs() << "Ctx.DeducedTemplates.size() != CP.DeducedTemplatesSize\n";
    eraseFoldingSetIf(Ctx.DeducedTemplates,
                      [&](DeducedTemplateStorage &Node) -> bool {
                        return Ctx.getAllocator().isAfterCheckpoint(
                            static_cast<void *>(&Node), SlabCP);
                      });
    assert(Ctx.DeducedTemplates.size() == CP.DeducedTemplatesSize);
  }

  if (Ctx.ArrayParameterTypes.size() != CP.ArrayParameterTypesSize) {
    llvm::dbgs()
        << "Ctx.ArrayParameterTypes.size() != CP.ArrayParameterTypesSize\n";
    eraseFoldingSetIf(Ctx.ArrayParameterTypes,
                      [&](ArrayParameterType &Node) -> bool {
                        return Ctx.getAllocator().isAfterCheckpoint(
                            static_cast<void *>(&Node), SlabCP);
                      });
    assert(Ctx.ArrayParameterTypes.size() == CP.ArrayParameterTypesSize);
  }

  if (CP.PredefinedSugarTypesSize != Ctx.PredefinedSugarTypes.size()) {
    llvm::dbgs() << "if (CP.PredefinedSugarTypesSize != "
                    "Ctx.PredefinedSugarTypes.size()\n";
    //                 mutable std::array<Type *,
    //                llvm::to_underlying(PredefinedSugarType::Kind::Last) +
    //                1>
    // PredefinedSugarTypes{};
    assert(CP.PredefinedSugarTypesSize == Ctx.PredefinedSugarTypes.size());
  }

  if (CP.NamespaceAndPrefixStoragesSize !=
      Ctx.NamespaceAndPrefixStorages.size()) {
    llvm::dbgs() << "if (CP.NamespaceAndPrefixStoragesSize != "
                    "Ctx.NamespaceAndPrefixStorages.size()\n";
    eraseFoldingSetIf(Ctx.NamespaceAndPrefixStorages,
                      [&](NamespaceAndPrefixStorage &Node) -> bool {
                        return Ctx.getAllocator().isAfterCheckpoint(
                            static_cast<void *>(&Node), SlabCP);
                      });
    assert(CP.NamespaceAndPrefixStoragesSize ==
           Ctx.NamespaceAndPrefixStorages.size());
  }

  if (CP.ASTRecordLayoutsSize != Ctx.ASTRecordLayouts.size()) {
    llvm::dbgs()
        << "if (CP.ASTRecordLayoutsSize != Ctx.ASTRecordLayouts.size()\n";
    eraseDenseMapIf(
        Ctx.ASTRecordLayouts,
        [&](llvm::detail::DenseMapPair<const RecordDecl *,
                                       const ASTRecordLayout *> &KV) -> bool {
          return Ctx.getAllocator().isAfterCheckpoint(
                     static_cast<void *>(
                         const_cast<RecordDecl *>(KV.getFirst())),
                     SlabCP) ||
                 Ctx.getAllocator().isAfterCheckpoint(
                     static_cast<void *>(
                         const_cast<ASTRecordLayout *>(KV.getSecond())),
                     SlabCP);
        });
    assert(CP.ASTRecordLayoutsSize == Ctx.ASTRecordLayouts.size());
  }

  if (CP.MemoizedTypeInfoSize != Ctx.MemoizedTypeInfo.size()) {
    llvm::dbgs()
        << "if (CP.MemoizedTypeInfoSize != Ctx.MemoizedTypeInfo.size() "
        << CP.MemoizedTypeInfoSize << " != " << Ctx.MemoizedTypeInfo.size()
        << " \n";
    eraseDenseMapIf(
        Ctx.MemoizedTypeInfo,
        [&](llvm::detail::DenseMapPair<const Type *, struct TypeInfo> &KV)
            -> bool {
          // llvm::outs() << "Type * In Range : [ "
          //              << static_cast<void *>(SlabCP.CurPtr) << " < "
          //              << KV.getFirst() << " < "
          //              << static_cast<void *>(SlabCP.End) << "]\n";
          return Ctx.getAllocator().isAfterCheckpoint(
              static_cast<void *>(const_cast<Type *>(KV.getFirst())), SlabCP);
        });
    // assert(CP.MemoizedTypeInfoSize == Ctx.MemoizedTypeInfo.size());
  }

  if (CP.MemoizedUnadjustedAlignSize != Ctx.MemoizedUnadjustedAlign.size()) {
    llvm::dbgs() << "if (CP.MemoizedUnadjustedAlignSize != "
                    "Ctx.MemoizedUnadjustedAlign.size()\n";
    eraseDenseMapIf(
        Ctx.MemoizedUnadjustedAlign,
        [&](llvm::detail::DenseMapPair<const Type *, unsigned> &KV) -> bool {
          return Ctx.getAllocator().isAfterCheckpoint(
              static_cast<void *>(const_cast<Type *>(KV.getFirst())), SlabCP);
        });
    assert(CP.MemoizedUnadjustedAlignSize ==
           Ctx.MemoizedUnadjustedAlign.size());
  }

  if (CP.KeyFunctionsSize != Ctx.KeyFunctions.size()) {
    llvm::dbgs() << "if (CP.KeyFunctionsSize != Ctx.KeyFunctions.size()\n";
    eraseDenseMapIf(
        Ctx.KeyFunctions,
        [&](llvm::detail::DenseMapPair<const CXXRecordDecl *, LazyDeclPtr> &KV)
            -> bool {
          return Ctx.getAllocator().isAfterCheckpoint(
              static_cast<void *>(const_cast<CXXRecordDecl *>(KV.getFirst())),
              SlabCP);
        });
    assert(CP.KeyFunctionsSize == Ctx.KeyFunctions.size());
  }

  if (CP.BlockVarCopyInitsSize != Ctx.BlockVarCopyInits.size()) {
    llvm::dbgs()
        << "if (CP.BlockVarCopyInitsSize != Ctx.BlockVarCopyInits.size()\n";
    eraseDenseMapIf(
        Ctx.BlockVarCopyInits,
        [&](llvm::detail::DenseMapPair<const VarDecl *, BlockVarCopyInit> &KV)
            -> bool {
          return Ctx.getAllocator().isAfterCheckpoint(
              static_cast<void *>(const_cast<VarDecl *>(KV.getFirst())),
              SlabCP);
        });
    assert(CP.BlockVarCopyInitsSize == Ctx.BlockVarCopyInits.size());
  }

  if (CP.MSGuidDeclsSize != Ctx.MSGuidDecls.size()) {
    llvm::dbgs() << "if (CP.MSGuidDeclsSize != Ctx.MSGuidDecls.size()\n";
    eraseFoldingSetIf(Ctx.MSGuidDecls, [&](MSGuidDecl &Node) -> bool {
      return Ctx.getAllocator().isAfterCheckpoint(static_cast<void *>(&Node),
                                                  SlabCP);
    });
    assert(CP.MSGuidDeclsSize == Ctx.MSGuidDecls.size());
  }

  if (CP.UnnamedGlobalConstantDeclsSize !=
      Ctx.UnnamedGlobalConstantDecls.size()) {
    llvm::dbgs() << "if (CP.UnnamedGlobalConstantDeclsSize != "
                    "Ctx.UnnamedGlobalConstantDecls.size()\n";
    eraseFoldingSetIf(Ctx.UnnamedGlobalConstantDecls,
                      [&](UnnamedGlobalConstantDecl &Node) -> bool {
                        return Ctx.getAllocator().isAfterCheckpoint(
                            static_cast<void *>(&Node), SlabCP);
                      });
    assert(CP.UnnamedGlobalConstantDeclsSize ==
           Ctx.UnnamedGlobalConstantDecls.size());
  }

  if (CP.TemplateParamObjectDeclsSize != Ctx.TemplateParamObjectDecls.size()) {
    llvm::dbgs() << "if (CP.TemplateParamObjectDeclsSize != "
                    "Ctx.TemplateParamObjectDecls.size()\n";
    eraseFoldingSetIf(Ctx.TemplateParamObjectDecls,
                      [&](TemplateParamObjectDecl &Node) -> bool {
                        return Ctx.getAllocator().isAfterCheckpoint(
                            static_cast<void *>(&Node), SlabCP);
                      });
    assert(CP.TemplateParamObjectDeclsSize ==
           Ctx.TemplateParamObjectDecls.size());
  }

  if (CP.StringLiteralCacheSize != Ctx.StringLiteralCache.size()) {
    llvm::dbgs()
        << "if (CP.StringLiteralCacheSize != Ctx.StringLiteralCache.size()\n";
    //   mutable llvm::StringMap<StringLiteral *> StringLiteralCache;
    assert(CP.StringLiteralCacheSize == Ctx.StringLiteralCache.size());
  }

  if (CP.DestroyingOperatorDeletesSize !=
      Ctx.DestroyingOperatorDeletes.size()) {
    llvm::dbgs() << "if (CP.DestroyingOperatorDeletesSize != "
                    "Ctx.DestroyingOperatorDeletes.size()\n";
    eraseDenseSetIf(
        Ctx.DestroyingOperatorDeletes, [&](const FunctionDecl *FD) -> bool {
          return Ctx.getAllocator().isAfterCheckpoint(
              static_cast<void *>(const_cast<FunctionDecl *>(FD)), SlabCP);
        });
    assert(CP.DestroyingOperatorDeletesSize ==
           Ctx.DestroyingOperatorDeletes.size());
  }

  if (CP.TypeAwareOperatorNewAndDeletesSize !=
      Ctx.TypeAwareOperatorNewAndDeletes.size()) {
    llvm::dbgs() << "CP.TypeAwareOperatorNewAndDeletesSize != "
                    "Ctx.TypeAwareOperatorNewAndDeletes.size()\n";
    eraseDenseSetIf(Ctx.TypeAwareOperatorNewAndDeletes,
                    [&](const FunctionDecl *FD) -> bool {
                      return Ctx.getAllocator().isAfterCheckpoint(
                          static_cast<void *>(const_cast<FunctionDecl *>(FD)),
                          SlabCP);
                    });
    assert(CP.TypeAwareOperatorNewAndDeletesSize ==
           Ctx.TypeAwareOperatorNewAndDeletes.size());
  }

  if (CP.OperatorDeletesForVirtualDtorSize !=
      Ctx.OperatorDeletesForVirtualDtor.size()) {
    llvm::dbgs() << "CP.OperatorDeletesForVirtualDtorSize != "
                    "Ctx.OperatorDeletesForVirtualDtor.size()\n";
    eraseDenseMapIf(
        Ctx.OperatorDeletesForVirtualDtor,
        [&](llvm::detail::DenseMapPair<const CXXDestructorDecl *,
                                       FunctionDecl *> &KV) -> bool {
          return Ctx.getAllocator().isAfterCheckpoint(
                     static_cast<void *>(
                         const_cast<CXXDestructorDecl *>(KV.getFirst())),
                     SlabCP) ||
                 Ctx.getAllocator().isAfterCheckpoint(
                     static_cast<void *>(KV.getSecond()), SlabCP);
        });
    assert(CP.OperatorDeletesForVirtualDtorSize ==
           Ctx.OperatorDeletesForVirtualDtor.size());
  }

  if (CP.GlobalOperatorDeletesForVirtualDtorSize !=
      Ctx.GlobalOperatorDeletesForVirtualDtor.size()) {
    llvm::dbgs() << "CP.GlobalOperatorDeletesForVirtualDtorSize != "
                    "Ctx.GlobalOperatorDeletesForVirtualDtor.size()\n";
    eraseDenseMapIf(
        Ctx.GlobalOperatorDeletesForVirtualDtor,
        [&](llvm::detail::DenseMapPair<const CXXDestructorDecl *,
                                       FunctionDecl *> &KV) -> bool {
          return Ctx.getAllocator().isAfterCheckpoint(
                     static_cast<void *>(
                         const_cast<CXXDestructorDecl *>(KV.getFirst())),
                     SlabCP) ||
                 Ctx.getAllocator().isAfterCheckpoint(
                     static_cast<void *>(KV.getSecond()), SlabCP);
        });
    assert(CP.GlobalOperatorDeletesForVirtualDtorSize ==
           Ctx.GlobalOperatorDeletesForVirtualDtor.size());
  }

  if (CP.ArrayOperatorDeletesForVirtualDtorSize !=
      Ctx.ArrayOperatorDeletesForVirtualDtor.size()) {
    llvm::dbgs() << "CP.ArrayOperatorDeletesForVirtualDtorSize != "
                    "Ctx.ArrayOperatorDeletesForVirtualDtor.size()\n";
    eraseDenseMapIf(
        Ctx.ArrayOperatorDeletesForVirtualDtor,
        [&](llvm::detail::DenseMapPair<const CXXDestructorDecl *,
                                       FunctionDecl *> &KV) -> bool {
          return Ctx.getAllocator().isAfterCheckpoint(
                     static_cast<void *>(
                         const_cast<CXXDestructorDecl *>(KV.getFirst())),
                     SlabCP) ||
                 Ctx.getAllocator().isAfterCheckpoint(
                     static_cast<void *>(KV.getSecond()), SlabCP);
        });
    assert(CP.ArrayOperatorDeletesForVirtualDtorSize ==
           Ctx.ArrayOperatorDeletesForVirtualDtor.size());
  }

  if (CP.GlobalArrayOperatorDeletesForVirtualDtorSize !=
      Ctx.GlobalArrayOperatorDeletesForVirtualDtor.size()) {
    llvm::dbgs() << "CP.GlobalArrayOperatorDeletesForVirtualDtorSize != "
                    "Ctx.GlobalArrayOperatorDeletesForVirtualDtor.size()\n";
    eraseDenseMapIf(
        Ctx.GlobalArrayOperatorDeletesForVirtualDtor,
        [&](llvm::detail::DenseMapPair<const CXXDestructorDecl *,
                                       FunctionDecl *> &KV) -> bool {
          return Ctx.getAllocator().isAfterCheckpoint(
                     static_cast<void *>(
                         const_cast<CXXDestructorDecl *>(KV.getFirst())),
                     SlabCP) ||
                 Ctx.getAllocator().isAfterCheckpoint(
                     static_cast<void *>(KV.getSecond()), SlabCP);
        });
    assert(CP.GlobalArrayOperatorDeletesForVirtualDtorSize ==
           Ctx.GlobalArrayOperatorDeletesForVirtualDtor.size());
  }

  if (CP.RequireVectorDeletingDtorSize !=
      Ctx.RequireVectorDeletingDtor.size()) {
    llvm::dbgs() << "if (CP.RequireVectorDeletingDtorSize != "
                    "Ctx.RequireVectorDeletingDtor.size()\n";
    eraseDenseSetIf(
        Ctx.RequireVectorDeletingDtor, [&](const CXXRecordDecl *RD) -> bool {
          return Ctx.getAllocator().isAfterCheckpoint(
              static_cast<void *>(const_cast<CXXRecordDecl *>(RD)), SlabCP);
        });
    assert(CP.RequireVectorDeletingDtorSize ==
           Ctx.RequireVectorDeletingDtor.size());
  }

  if (CP.MergedDeclsSize != Ctx.MergedDecls.size()) {
    llvm::dbgs() << "CP.MergedDeclsSize != Ctx.MergedDecls.size()";
    assert(CP.MergedDeclsSize == Ctx.MergedDecls.size());
  }

  if (CP.MergedDefModulesSize != Ctx.MergedDefModules.size()) {
    llvm::dbgs() << "CP.MergedDefModulesSize != Ctx.MergedDefModules.size()";
    assert(CP.MergedDefModulesSize == Ctx.MergedDefModules.size());
  }

  if (CP.ModuleInitializersSize != Ctx.ModuleInitializers.size()) {
    llvm::dbgs()
        << "CP.ModuleInitializersSize != Ctx.ModuleInitializers.size()";
    assert(CP.ModuleInitializersSize == Ctx.ModuleInitializers.size());
  }

  if (CP.PrimaryModuleNameMapSize != Ctx.PrimaryModuleNameMap.size()) {
    llvm::dbgs()
        << "CP.PrimaryModuleNameMapSize != Ctx.PrimaryModuleNameMap.size()";
    assert(CP.PrimaryModuleNameMapSize == Ctx.PrimaryModuleNameMap.size());
  }

  if (CP.SameModuleLookupSetSize != Ctx.SameModuleLookupSet.size()) {
    llvm::dbgs()
        << "CP.SameModuleLookupSetSize != Ctx.SameModuleLookupSet.size()";
    assert(CP.SameModuleLookupSetSize == Ctx.SameModuleLookupSet.size());
  }

  if (CP.ScalableVecTyMapSize != Ctx.ScalableVecTyMap.size()) {
    llvm::dbgs() << "CP.ScalableVecTyMapSize != Ctx.ScalableVecTyMap.size()";
    assert(CP.ScalableVecTyMapSize == Ctx.ScalableVecTyMap.size());
  }

  if (CP.LambdaCastPathsSize != Ctx.LambdaCastPaths.size()) {
    llvm::dbgs() << "CP.LambdaCastPathsSize != Ctx.LambdaCastPaths.size()";
    assert(CP.LambdaCastPathsSize == Ctx.LambdaCastPaths.size());
  }

  if (CP.DeclRawCommentsSize != Ctx.DeclRawComments.size()) {
    llvm::dbgs() << "CP.DeclRawCommentsSize != Ctx.DeclRawComments.size()";
    assert(CP.DeclRawCommentsSize == Ctx.DeclRawComments.size());
  }

  if (CP.RedeclChainCommentsSize != Ctx.RedeclChainComments.size()) {
    llvm::dbgs()
        << "CP.RedeclChainCommentsSize != Ctx.RedeclChainComments.size()";
    assert(CP.RedeclChainCommentsSize == Ctx.RedeclChainComments.size());
  }

  if (CP.CommentlessRedeclChainsSize != Ctx.CommentlessRedeclChains.size()) {
    llvm::dbgs() << "CP.CommentlessRedeclChainsSize != "
                    "Ctx.CommentlessRedeclChains.size()";
    assert(CP.CommentlessRedeclChainsSize ==
           Ctx.CommentlessRedeclChains.size());
  }

  if (CP.ParsedCommentsSize != Ctx.ParsedComments.size()) {
    llvm::dbgs() << "CP.ParsedCommentsSize != Ctx.ParsedComments.size()";
    assert(CP.ParsedCommentsSize == Ctx.ParsedComments.size());
  }

  if (CP.RelocatableClassesSize != Ctx.RelocatableClasses.size()) {
    llvm::dbgs()
        << "CP.RelocatableClassesSize != Ctx.RelocatableClasses.size()";
    assert(CP.RelocatableClassesSize == Ctx.RelocatableClasses.size());
  }

  if (CP.ParamIndicesSize != Ctx.ParamIndices.size()) {
    llvm::dbgs() << "CP.ParamIndicesSize != Ctx.ParamIndices.size()";
    assert(CP.ParamIndicesSize == Ctx.ParamIndices.size());
  }

  if (CP.MangleNumbersSize != Ctx.MangleNumbers.size()) {
    llvm::dbgs() << "CP.MangleNumbersSize != Ctx.MangleNumbers.size()";
    assert(CP.MangleNumbersSize == Ctx.MangleNumbers.size());
  }

  if (CP.StaticLocalNumbersSize != Ctx.StaticLocalNumbers.size()) {
    llvm::dbgs()
        << "CP.StaticLocalNumbersSize != Ctx.StaticLocalNumbers.size()";
    assert(CP.StaticLocalNumbersSize == Ctx.StaticLocalNumbers.size());
  }

  if (CP.TemplateOrInstantiationSize != Ctx.TemplateOrInstantiation.size()) {
    llvm::dbgs() << "CP.TemplateOrInstantiationSize != "
                    "Ctx.TemplateOrInstantiation.size()";
    assert(CP.TemplateOrInstantiationSize ==
           Ctx.TemplateOrInstantiation.size());
  }

  if (CP.InstantiatedFromUsingDeclSize !=
      Ctx.InstantiatedFromUsingDecl.size()) {
    llvm::dbgs() << "CP.InstantiatedFromUsingDeclSize != "
                    "Ctx.InstantiatedFromUsingDecl.size()";
    assert(CP.InstantiatedFromUsingDeclSize ==
           Ctx.InstantiatedFromUsingDecl.size());
  }

  if (CP.InstantiatedFromUsingEnumDeclSize !=
      Ctx.InstantiatedFromUsingEnumDecl.size()) {
    llvm::dbgs() << "CP.InstantiatedFromUsingEnumDeclSize != "
                    "Ctx.InstantiatedFromUsingEnumDecl.size()";
    assert(CP.InstantiatedFromUsingEnumDeclSize ==
           Ctx.InstantiatedFromUsingEnumDecl.size());
  }

  if (CP.InstantiatedFromUsingShadowDeclSize !=
      Ctx.InstantiatedFromUsingShadowDecl.size()) {
    llvm::dbgs() << "CP.InstantiatedFromUsingShadowDeclSize != "
                    "Ctx.InstantiatedFromUsingShadowDecl.size()";
    assert(CP.InstantiatedFromUsingShadowDeclSize ==
           Ctx.InstantiatedFromUsingShadowDecl.size());
  }

  if (CP.InstantiatedFromUnnamedFieldDeclSize !=
      Ctx.InstantiatedFromUnnamedFieldDecl.size()) {
    llvm::dbgs() << "CP.InstantiatedFromUnnamedFieldDeclSize != "
                    "Ctx.InstantiatedFromUnnamedFieldDecl.size()";
    assert(CP.InstantiatedFromUnnamedFieldDeclSize ==
           Ctx.InstantiatedFromUnnamedFieldDecl.size());
  }

  if (CP.OverriddenMethodsSize != Ctx.OverriddenMethods.size()) {
    llvm::dbgs() << "CP.OverriddenMethodsSize != Ctx.OverriddenMethods.size()";
    assert(CP.OverriddenMethodsSize == Ctx.OverriddenMethods.size());
  }

  if (CP.MangleNumberingContextsSize != Ctx.MangleNumberingContexts.size()) {
    llvm::dbgs() << "CP.MangleNumberingContextsSize != "
                    "Ctx.MangleNumberingContexts.size()";
    assert(CP.MangleNumberingContextsSize ==
           Ctx.MangleNumberingContexts.size());
  }

  if (CP.ExtraMangleNumberingContextsSize !=
      Ctx.ExtraMangleNumberingContexts.size()) {
    llvm::dbgs() << "CP.ExtraMangleNumberingContextsSize != "
                    "Ctx.ExtraMangleNumberingContexts.size()";
    assert(CP.ExtraMangleNumberingContextsSize ==
           Ctx.ExtraMangleNumberingContexts.size());
  }

  if (CP.TraversalScopeSize != Ctx.TraversalScope.size()) {
    llvm::dbgs() << "CP.TraversalScopeSize != "
                    "Ctx.TraversalScope.size()";
    assert(CP.TraversalScopeSize == Ctx.TraversalScope.size());
  }

  for (auto &[Decl, OldTy] : Ctx.PendingTypeForDeclMutations)
    Decl->TypeForDecl = OldTy;

  Ctx.PendingTypeForDeclMutations.clear();
  TranslationUnitDecl *MostRecentTU = Ctx.getTranslationUnitDecl();
  for (const auto *DC : Ctx.PendingDCMutations) {
    if (MostRecentTU->getPrimaryContext() == DC)
      continue;
    if (StoredDeclsMap *Map = const_cast<DeclContext *>(DC)
                                  ->getPrimaryContext()
                                  ->getLookupPtr()) {
      for (auto &&[Key, List] : *Map) {
        DeclContextLookupResult R = List.getLookupResult();
        std::vector<NamedDecl *> NamedDeclsToRemove;
        // bool RemoveAll = true;
        for (NamedDecl *D : R) {
          // llvm::outs() << "D->getTranslationUnitDecl() == MostRecentTU (" << (D->getTranslationUnitDecl() == MostRecentTU) << ")\n";
          // llvm::outs() << "DeclContext : " << DC << "\n";
          // D->dump();
          // if (D->getTranslationUnitDecl() == MostRecentTU)
          if (Ctx.getAllocator().isAfterCheckpoint(static_cast<void *>(D),
                                                   SlabCP))
            NamedDeclsToRemove.push_back(D);
          // else
          //   RemoveAll = false;
        }
        // if (LLVM_LIKELY(RemoveAll)) {
        //   Map->erase(Key);
        // } else {
        for (NamedDecl *D : NamedDeclsToRemove)
          List.remove(D);
        // }
      }
    }

    Decl *Prev = DC->FirstDecl;
    Decl *Cur = DC->FirstDecl;
    while (Cur) {
      if (Ctx.getAllocator().isAfterCheckpoint(static_cast<void *>(Cur),
                                               SlabCP)) {
        DC->LastDecl = Prev;
        DC->LastDecl->NextInContextAndBits.setPointer(nullptr);
        break;
      }
      Prev = Cur;
      Cur = Cur->getNextDeclInContext();
    }
  }

  // if (FirstDecl) {
  //   LastDecl->NextInContextAndBits.setPointer(D);
  //   LastDecl = D;
  // } else {
  //   FirstDecl = LastDecl = D;
  // }

  // Notify a C++ record declaration that we've added a member, so it can
  // update its class-specific state.
  // if (auto *Record = dyn_cast<CXXRecordDecl>(this))
  //   Record->addedMember(D);


  //   ImportDecl *FirstLocalImport = nullptr;
  // ImportDecl *LastLocalImport = nullptr;


  // class ImportDecl final : public Decl,
  //                        llvm::TrailingObjects<ImportDecl, SourceLocation> {
  // friend class ASTContext;
  // friend class ASTDeclReader;
  // friend class ASTReader;
  // friend TrailingObjects;

  // /// The imported module.
  // Module *ImportedModule = nullptr;

  // /// The next import in the list of imports local to the translation
  // /// unit being parsed (not loaded from an AST file).
  // ///
  // /// Includes a bit that indicates whether we have source-location information
  // /// for each identifier in the module name.
  // ///
  // /// When the bit is false, we only have a single source location for the
  // /// end of the import declaration.
  // llvm::PointerIntPair<ImportDecl *, 1, bool> NextLocalImportAndComplete;

  Ctx.PendingDCMutations.clear();

  llvm::PointerIntPair<StoredDeclsMap*,1> LastSDM = Ctx.LastSDM;

  StoredDeclsMap *Map = LastSDM.getPointer();
  bool Dependent = LastSDM.getInt();
  while (Map && (Map != CP.LastSDM.getPointer())) {
    // llvm::outs() << "we are deleting LASTSDM\n";
  //   // Advance the iteration before we invalidate memory.
    llvm::PointerIntPair<StoredDeclsMap*,1> Next = Map->Previous;

    if (Dependent)
      delete static_cast<DependentStoredDeclsMap*>(Map);
    else
      delete Map;

    Map = Next.getPointer();
    Dependent = Next.getInt();
  }

  Ctx.LastSDM = CP.LastSDM;

  Ctx.Types.resize(CP.TypesSize);
}

void ASTContextStateStash::commit() {
  Ctx.PendingTypeForDeclMutations.clear();
  Ctx.PendingDCMutations.clear();
}
} // end namespace clang