//===-- DebugTypeGenerator.h -- type conversion ------------------- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Coding style: https://mlir.llvm.org/getting_started/DeveloperGuide/
//
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_OPTIMIZER_TRANSFORMS_DEBUGTYPEGENERATOR_H
#define FORTRAN_OPTIMIZER_TRANSFORMS_DEBUGTYPEGENERATOR_H

#include "flang/Optimizer/CodeGen/TypeConverter.h"
#include "flang/Optimizer/Dialect/FIRCG/CGOps.h"
#include "flang/Optimizer/Dialect/FIRType.h"
#include "flang/Optimizer/Dialect/Support/FIRContext.h"
#include "flang/Optimizer/Dialect/Support/KindMapping.h"
#include "flang/Optimizer/Support/DataLayout.h"
#include "llvm/Support/Debug.h"

namespace fir {

/// Special cache to deal with the fact that mlir::LLVM::DITypeAttr for
/// derived types may only be valid in specific nesting contexts in presence
/// of derived type recursion and cannot be cached for the whole compilation.
/// It is however still desirable to cache such mlir::LLVM::DITypeAttr as
/// long as possible to avoid catastrophic compilation slow downs in very
/// complex derived types where an intermediate type in a derived type cycle may
/// indirectly appear hundreds of times under the top type of the derived type
/// cycle. More details in the comment below.
class DerivedTypeCache {
public:
  // Currently, the handling of recursive debug type in mlir has some
  // limitations that were discussed at the end of the thread for following
  // PR.
  // https://github.com/llvm/llvm-project/pull/106571
  //
  // Problem could be explained with the following example code:
  //  type t2
  //   type(t1), pointer :: p1
  // end type
  // type t1
  //   type(t2), pointer :: p2
  // end type
  // In the description below, type_self means a temporary type that is
  // generated
  // as a place holder while the members of that type are being processed.
  //
  // If we process t1 first then we will have the following structure after
  // it has been processed.
  // t1 -> t2 -> t1_self
  // This is because when we started processing t2, we did not have the
  // complete t1 but its place holder t1_self.
  // Now if some entity requires t2, we will already have that in cache and
  // will return it. But this t2 refers to t1_self and not to t1. In mlir
  // handling, only those types are allowed to have _self reference which are
  // wrapped by entity whose reference it is. So t1 -> t2 -> t1_self is ok
  // because the t1_self reference can be resolved by the outer t1. But
  // standalone t2 is not because there will be no way to resolve it. Until
  // this is fixed in mlir, we avoid caching such types. Please see
  // DebugTranslation::translateRecursive for details on how mlir handles
  // recursive types.
  using ActiveLevels = llvm::SmallVector<int32_t, 1>;
  mlir::LLVM::DITypeAttr lookup(mlir::Type);
  ActiveLevels startTranslating(mlir::Type,
                                mlir::LLVM::DITypeAttr placeHolder = nullptr);
  void finalize(mlir::Type, mlir::LLVM::DITypeAttr, ActiveLevels &&);
  void preComponentVisitUpdate();
  void postComponentVisitUpdate(ActiveLevels &);

private:
  void insertCacheCleanUp(mlir::Type type, int32_t depth);
  void cleanUpCache(int32_t depth);
  // Current depth inside a top level derived type being converted.
  int32_t derivedTypeDepth = 0;
  // Cache for already translated derived types with the minimum depth where
  // this cache entry is valid. Zero means the translation is always valid, "i"
  // means the type depends its derived type tree parent node at depth "i". Such
  // types should be cleaned-up from the cache in the post visit of node "i".
  // Note that any new metadata created for a type with a component in the cache
  // with validity of "i" shall not be added to the cache with a validity
  // smaller than "i".
  llvm::DenseMap<mlir::Type, std::pair<mlir::LLVM::DITypeAttr, ActiveLevels>>
      typeCache;
  // List of parent nodes that are being recursively referred to in the
  // component type that has just been computed.
  ActiveLevels componentActiveRecursionLevels;
  // Helper list that maintains the list of nodes that must be deleted from the
  // cache when going back past listed parent depths.
  llvm::SmallVector<std::pair<llvm::SmallVector<mlir::Type>, int32_t>>
      cacheCleanupList;
};

/// This converts FIR/mlir type to DITypeAttr.
class DebugTypeGenerator {
public:
  DebugTypeGenerator(mlir::ModuleOp module, mlir::SymbolTable *symbolTable,
                     const mlir::DataLayout &dl);

  mlir::LLVM::DITypeAttr convertType(mlir::Type Ty,
                                     mlir::LLVM::DIFileAttr fileAttr,
                                     mlir::LLVM::DIScopeAttr scope,
                                     fir::cg::XDeclareOp declOp);

private:
  mlir::LLVM::DITypeAttr convertRecordType(fir::RecordType Ty,
                                           mlir::LLVM::DIFileAttr fileAttr,
                                           mlir::LLVM::DIScopeAttr scope,
                                           fir::cg::XDeclareOp declOp);
  mlir::LLVM::DITypeAttr convertTupleType(mlir::TupleType Ty,
                                          mlir::LLVM::DIFileAttr fileAttr,
                                          mlir::LLVM::DIScopeAttr scope,
                                          fir::cg::XDeclareOp declOp);
  mlir::LLVM::DITypeAttr convertSequenceType(fir::SequenceType seqTy,
                                             mlir::LLVM::DIFileAttr fileAttr,
                                             mlir::LLVM::DIScopeAttr scope,
                                             fir::cg::XDeclareOp declOp);
  mlir::LLVM::DITypeAttr convertVectorType(fir::VectorType vecTy,
                                           mlir::LLVM::DIFileAttr fileAttr,
                                           mlir::LLVM::DIScopeAttr scope,
                                           fir::cg::XDeclareOp declOp);

  /// The 'genAllocated' is true when we want to generate 'allocated' field
  /// in the DICompositeType. It is needed for the allocatable arrays.
  /// Similarly, 'genAssociated' is used with 'pointer' type to generate
  /// 'associated' field.
  mlir::LLVM::DITypeAttr convertBoxedSequenceType(
      fir::SequenceType seqTy, mlir::LLVM::DIFileAttr fileAttr,
      mlir::LLVM::DIScopeAttr scope, fir::cg::XDeclareOp declOp,
      bool genAllocated, bool genAssociated);
  mlir::LLVM::DITypeAttr convertCharacterType(fir::CharacterType charTy,
                                              mlir::LLVM::DIFileAttr fileAttr,
                                              mlir::LLVM::DIScopeAttr scope,
                                              fir::cg::XDeclareOp declOp,
                                              bool hasDescriptor);

  mlir::LLVM::DITypeAttr convertPointerLikeType(mlir::Type elTy,
                                                mlir::LLVM::DIFileAttr fileAttr,
                                                mlir::LLVM::DIScopeAttr scope,
                                                fir::cg::XDeclareOp declOp,
                                                bool genAllocated,
                                                bool genAssociated);
  mlir::LLVM::DILocalVariableAttr
  generateArtificialVariable(mlir::MLIRContext *context, mlir::Value Val,
                             mlir::LLVM::DIFileAttr fileAttr,
                             mlir::LLVM::DIScopeAttr scope,
                             fir::cg::XDeclareOp declOp);
  std::pair<std::uint64_t, unsigned short>
  getFieldSizeAndAlign(mlir::Type fieldTy);

  mlir::ModuleOp module;
  mlir::SymbolTable *symbolTable;
  const mlir::DataLayout *dataLayout;
  KindMapping kindMapping;
  fir::LLVMTypeConverter llvmTypeConverter;
  std::uint64_t dimsSize;
  std::uint64_t dimsOffset;
  std::uint64_t ptrSize;
  std::uint64_t lenOffset;
  std::uint64_t rankOffset;
  std::uint64_t rankSize;
  DerivedTypeCache derivedTypeCache;
};

} // namespace fir

static uint32_t getLineFromLoc(mlir::Location loc) {
  uint32_t line = 1;
  if (auto fileLoc = mlir::dyn_cast<mlir::FileLineColLoc>(loc))
    line = fileLoc.getLine();
  return line;
}

#endif // FORTRAN_OPTIMIZER_TRANSFORMS_DEBUGTYPEGENERATOR_H
