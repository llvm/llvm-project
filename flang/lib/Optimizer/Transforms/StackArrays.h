//===- StackArrays.h ------------------------------------------------------===//
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
//
// This header exposes the reusable pieces of the StackArrays pass: the
// analysis that determines which fir.allocmem operations can safely be moved
// to the stack (and where the replacement fir.alloca should be inserted), and
// the pattern that performs the heap-to-stack rewrite. They are shared so that
// other passes can reuse the same "is this heap allocation safely stackifiable,
// and where" logic.
//
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_OPTIMIZER_TRANSFORMS_STACKARRAYS_H
#define FORTRAN_OPTIMIZER_TRANSFORMS_STACKARRAYS_H

#include "flang/Optimizer/Dialect/FIROps.h"
#include "flang/Optimizer/Dialect/Support/KindMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/DataLayoutInterfaces.h"
#include "mlir/Support/TypeID.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/PointerUnion.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include <optional>

namespace fir {

/// Return true if a fir.allocmem of \p type may be moved to the stack: arrays,
/// and the scalar characters that character expressions allocate for their
/// results. Like dynamic-size arrays, dynamic-length characters are only moved
/// where their stack storage is reclaimed.
inline bool isStackArraysCandidateType(mlir::Type type) {
  return mlir::isa<fir::SequenceType, fir::CharacterType>(type);
}

/// Stack scopes of a block in which the allocations that must stay inside a
/// loop share one llvm.stacksave/llvm.stackrestore pair.
struct BlockStackScopes {
  /// Whether a stack scope covering the whole block can be added without
  /// reclaiming stack storage that is still in use.
  bool canSaveRestore = false;
  /// The llvm.stacksave/llvm.stackrestore pairs already in the block.
  llvm::SmallVector<std::pair<mlir::Operation *, mlir::Operation *>> pairs;
};
using BlockStackScopesMap = llvm::DenseMap<mlir::Block *, BlockStackScopes>;

/// Stores where an alloca should be inserted. If the PointerUnion is an
/// Operation the alloca should be inserted /after/ the operation. If it is a
/// block, the alloca can be placed anywhere in that block.
class InsertionPoint {
  llvm::PointerUnion<mlir::Operation *, mlir::Block *> location;
  bool saveRestoreStack;

  /// Get contained pointer type or nullptr
  template <class T>
  T *tryGetPtr() const {
    // Use llvm::dyn_cast_if_present because location may be null here.
    if (T *ptr = llvm::dyn_cast_if_present<T *>(location))
      return ptr;
    return nullptr;
  }

public:
  template <class T>
  InsertionPoint(T *ptr, bool saveRestoreStack = false)
      : location(ptr), saveRestoreStack{saveRestoreStack} {}
  InsertionPoint(std::nullptr_t null)
      : location(null), saveRestoreStack{false} {}

  /// Get contained operation, or nullptr
  mlir::Operation *tryGetOperation() const {
    return tryGetPtr<mlir::Operation>();
  }

  /// Get contained block, or nullptr
  mlir::Block *tryGetBlock() const { return tryGetPtr<mlir::Block>(); }

  /// Get whether the stack should be saved/restored. If yes, an llvm.stacksave
  /// intrinsic should be added at the start of the allocation block, and an
  /// llvm.stackrestore intrinsic should be added before its terminator.
  bool shouldSaveRestoreStack() const { return saveRestoreStack; }

  operator bool() const { return tryGetOperation() || tryGetBlock(); }

  bool operator==(const InsertionPoint &rhs) const {
    return (location == rhs.location) &&
           (saveRestoreStack == rhs.saveRestoreStack);
  }

  bool operator!=(const InsertionPoint &rhs) const { return !(*this == rhs); }
};

/// Drives analysis to find candidate fir.allocmem operations which could be
/// moved to the stack. Intended to be used with mlir::Pass::getAnalysis
class StackArraysAnalysisWrapper {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(StackArraysAnalysisWrapper)

  // Maps fir.allocmem -> place to insert alloca
  using AllocMemMap = llvm::DenseMap<mlir::Operation *, InsertionPoint>;

  StackArraysAnalysisWrapper(mlir::Operation *op) {}

  // Returns nullptr if analysis failed.
  // Note: the returned pointer points into funcMaps and is invalidated if
  // funcMaps grows (i.e. when a not-yet-analysed function is queried). This
  // does not happen currently because each StackArraysAnalysisWrapper instance
  // is only used to analyse a single function.
  const AllocMemMap *getCandidateOps(mlir::Operation *func);

private:
  llvm::DenseMap<mlir::Operation *, AllocMemMap> funcMaps;

  llvm::LogicalResult analyseFunction(mlir::Operation *func);
};

/// Converts a fir.allocmem to a fir.alloca
class AllocMemConversion : public mlir::OpRewritePattern<fir::AllocMemOp> {
public:
  explicit AllocMemConversion(
      mlir::MLIRContext *ctx,
      const StackArraysAnalysisWrapper::AllocMemMap &candidateOps,
      std::optional<mlir::DataLayout> &dl,
      std::optional<fir::KindMapping> &kindMap)
      : OpRewritePattern(ctx), candidateOps{candidateOps}, dl{dl},
        kindMap{kindMap} {}

  llvm::LogicalResult
  matchAndRewrite(fir::AllocMemOp allocmem,
                  mlir::PatternRewriter &rewriter) const override;

  /// Determine where to insert the alloca operation. The returned value should
  /// be checked to see if it is inside a loop
  static InsertionPoint
  findAllocaInsertionPoint(fir::AllocMemOp &oldAlloc,
                           const llvm::SmallVector<mlir::Operation *> &freeOps,
                           BlockStackScopesMap &blockStackScopes);

private:
  /// Handle to the DFA (already run)
  const StackArraysAnalysisWrapper::AllocMemMap &candidateOps;

  const std::optional<mlir::DataLayout> &dl;
  const std::optional<fir::KindMapping> &kindMap;
  mutable llvm::DenseSet<mlir::Block *> blocksWithStackSaveRestore;

  /// If we failed to find an insertion point not inside a loop, see if it would
  /// be safe to use an llvm.stacksave/llvm.stackrestore inside the loop
  static InsertionPoint findAllocaLoopInsertionPoint(
      fir::AllocMemOp &oldAlloc,
      const llvm::SmallVector<mlir::Operation *> &freeOps,
      BlockStackScopesMap &blockStackScopes);

  /// Returns the alloca if it was successfully inserted, otherwise {}
  std::optional<fir::AllocaOp>
  insertAlloca(fir::AllocMemOp &oldAlloc,
               mlir::PatternRewriter &rewriter) const;

  /// Insert one stacksave/stackrestore pair for the entire allocation block.
  void insertStackSaveRestore(fir::AllocMemOp oldAlloc,
                              mlir::PatternRewriter &rewriter) const;
  /// Emit lifetime markers for newAlloc between oldAlloc and each freemem.
  /// If the allocation is dynamic, no life markers are emitted.
  void insertLifetimeMarkers(fir::AllocMemOp oldAlloc, fir::AllocaOp newAlloc,
                             mlir::PatternRewriter &rewriter) const;
};

} // namespace fir

#endif // FORTRAN_OPTIMIZER_TRANSFORMS_STACKARRAYS_H
