//===- StmtRipple.h - Classes for Ripple directives  ----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// This file defines Ripple AST classes for statement-level constructs.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_AST_STMTRIPPLE_H
#define LLVM_CLANG_AST_STMTRIPPLE_H

#include "clang/AST/Decl.h"
#include "clang/AST/Expr.h"
#include "clang/AST/Stmt.h"
#include "clang/AST/StmtIterator.h"
#include "clang/Basic/LLVM.h"
#include "clang/Basic/SourceLocation.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/TrailingObjects.h"
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>

namespace clang {

class ASTContext;

/// This class represents a compute construct, representing in C or C++ a
/// `parallel' loop statement in canonical form
class RippleComputeConstruct final
    : public Stmt,
      public llvm::TrailingObjects<RippleComputeConstruct, uint64_t> {
  friend class ASTStmtWriter;
  friend class ASTStmtReader;
  friend class ASTContext;
  // The range of the pragma
  SourceRange Range;
  // The range of the PE directive
  SourceRange PERange;
  // The range of the Dims
  SourceRange DimsRange;

  ValueDecl *BlockShape;
  size_t NumDimensionIds;
  bool NoRemainder = false;

  /// Children of this AST node.
  enum {
    LOOP_STMT,
    PARALLEL_BLOCK_SIZE,
    DISTANCE_EXPR,
    DISTANCE_RIPPLE_FULL_BLOCK,
    LB_INIT_RIPPLE,
    STEP_RIPPLE,
    LOOP_IV_ORIGIN,
    LOOP_INIT_RIPPLE,
    LOOP_IV_UPDATE,
    RIPPLE_LOOP_STMT,
    REMAINDER_RUNTIME_COND,
    LOOP_IV_SET_TO_UB,
    REMAINDER_BODY,
    LastSubStmt = REMAINDER_BODY,
    FirstVarDecl = PARALLEL_BLOCK_SIZE,
    LastVarDecl = LOOP_INIT_RIPPLE,
    NumVarDecls = LastVarDecl - FirstVarDecl + 1
  };

  Stmt *SubStmts[LastSubStmt + 1] = {};

  RippleComputeConstruct(uint64_t NumDims)
      : Stmt(RippleComputeConstructClass), NumDimensionIds(NumDims) {}

  RippleComputeConstruct(SourceRange PragmaRange, SourceRange PERange,
                         SourceRange DimsRange, ValueDecl *BlockShape,
                         ArrayRef<uint64_t> Dims, ForStmt *AssociatedLoop,
                         bool NoRemainder)
      : Stmt(RippleComputeConstructClass), Range(PragmaRange), PERange(PERange),
        DimsRange(DimsRange), BlockShape(BlockShape),
        NumDimensionIds(Dims.size()), NoRemainder(NoRemainder) {
    setAssociatedForStmt(AssociatedLoop);
    std::uninitialized_copy(Dims.begin(), Dims.end(),
                            getTrailingObjectsNonStrict<uint64_t>());
  }

public:
  void printPragma(raw_ostream &OS) const;
  void print(raw_ostream &OS) const;

  static bool classof(const Stmt *T) {
    return T->getStmtClass() == RippleComputeConstructClass;
  }

  static RippleComputeConstruct *
  Create(const ASTContext &C, SourceRange PragmaLoc, SourceRange PELoc,
         SourceRange DimsLoc, ValueDecl *BlockShape, ArrayRef<uint64_t> Dims,
         ForStmt *AssociatedLoop, bool NoRemainder);
  static RippleComputeConstruct *CreateEmpty(const ASTContext &C,
                                             uint64_t NumDims);

  SourceLocation getBeginLoc() const { return Range.getBegin(); }
  SourceLocation getEndLoc() const { return Range.getEnd(); }
  SourceRange getPragmaRange() const { return Range; }
  SourceRange getProcessingElementRange() const { return PERange; }
  SourceRange getDimsRange() const { return DimsRange; }

  ValueDecl *getBlockShape() const { return BlockShape; }
  ArrayRef<uint64_t> getDimensionIds() const {
    return ArrayRef<uint64_t>(getTrailingObjectsNonStrict<uint64_t>(),
                              NumDimensionIds);
  }
  MutableArrayRef<uint64_t> dimensionIds() {
    return MutableArrayRef<uint64_t>(getTrailingObjectsNonStrict<uint64_t>(),
                                     NumDimensionIds);
  }

  // Accessor for the VarDecls defined by this construct
  SmallVector<const VarDecl *, NumVarDecls> getRippleVarDecls() const;

  // True if we need to generate the remainder loop section, false otherwise
  bool generateRemainder() const { return !NoRemainder; }

  /// The associated for statement
  ForStmt *getAssociatedForStmt() {
    return cast_if_present<ForStmt>(SubStmts[LOOP_STMT]);
  }
  const ForStmt *getAssociatedForStmt() const {
    return const_cast<RippleComputeConstruct *>(this)->getAssociatedForStmt();
  }
  void setAssociatedForStmt(ForStmt *For) {
    SubStmts[LOOP_STMT] = cast_if_present<Stmt>(For);
  }

  child_range children() {
    return child_range(SubStmts, SubStmts + LastSubStmt + 1);
  }
  const_child_range children() const {
    return const_cast<RippleComputeConstruct *>(this)->children();
  }

  // The size of the ripple parallel block
  Expr *getParallelBlockSize() const {
    return cast_if_present<Expr>(SubStmts[PARALLEL_BLOCK_SIZE]);
  }
  void setParallelBlockSize(Expr *E) {
    SubStmts[PARALLEL_BLOCK_SIZE] = cast_if_present<Stmt>(E);
  }

  // The number of iterations of the associated loop
  Expr *getAssociatedLoopIters() const {
    return cast_if_present<Expr>(SubStmts[DISTANCE_EXPR]);
  }
  void setAssociatedLoopIters(Expr *E) {
    SubStmts[DISTANCE_EXPR] = cast_if_present<Stmt>(E);
  }

  /// The number of full block iterations of the Ripple loop
  Expr *getRippleFullBlockIters() const {
    return cast_if_present<Expr>(SubStmts[DISTANCE_RIPPLE_FULL_BLOCK]);
  }
  void setRippleFullBlockIters(Expr *E) {
    SubStmts[DISTANCE_RIPPLE_FULL_BLOCK] = cast_if_present<Stmt>(E);
  }

  /// The new init for the original loop
  /// LoopInit + sum(ripple_id(dimensionIds()) * inner_dim_size)
  Expr *getRippleLowerBound() const {
    return cast_if_present<Expr>(SubStmts[LB_INIT_RIPPLE]);
  }
  void setRippleLowerBound(Expr *E) {
    SubStmts[LB_INIT_RIPPLE] = cast_if_present<Stmt>(E);
  }

  /// The new step of the original loop
  /// LoopStep + product(ripple_get_block_size(dimensionIds()))
  Expr *getRippleStep() const {
    return cast_if_present<Expr>(SubStmts[STEP_RIPPLE]);
  }
  void setRippleStep(Expr *E) {
    SubStmts[STEP_RIPPLE] = cast_if_present<Stmt>(E);
  }

  // RippleIV is used to generate the loop over full iterations:
  // for (RippleIV = 0; RippleIV < FullIterationCount; RippleIV += 1)
  Expr *getLoopInit() const {
    return cast_if_present<Expr>(SubStmts[LOOP_INIT_RIPPLE]);
  }
  void setLoopInit(Expr *E) {
    SubStmts[LOOP_INIT_RIPPLE] = cast_if_present<Stmt>(E);
  }

  /// The associated for statement's induction variable update
  /// LoopIV = RippleIndex + LoopInit +- (LoopStep * RippleSize) * RippleIV
  Expr *getLoopIVUpdate() const {
    return cast_if_present<Expr>(SubStmts[LOOP_IV_UPDATE]);
  }
  void setLoopIVUpdate(Expr *E) {
    SubStmts[LOOP_IV_UPDATE] = cast_if_present<Stmt>(E);
  }

  /// The crafted ripple loop
  ForStmt *getRippleLoopStmt() const {
    return cast_if_present<ForStmt>(SubStmts[RIPPLE_LOOP_STMT]);
  }
  void setRippleLoopStmt(ForStmt *For) {
    SubStmts[RIPPLE_LOOP_STMT] = cast_if_present<Stmt>(For);
  }

  // Runtime condition that evaluates to true when we need to execute the
  // remainder loop iteration (Masked section)
  Expr *getRemainderRuntimeCond() const {
    return cast_if_present<Expr>(SubStmts[REMAINDER_RUNTIME_COND]);
  }
  void setRemainderRuntimeCond(Expr *E) {
    SubStmts[REMAINDER_RUNTIME_COND] = cast_if_present<Stmt>(E);
  }

  /// Remainder body (loop body for the last masked iteration)
  Stmt *getRemainderBody() const { return SubStmts[REMAINDER_BODY]; }
  void setRemainderBody(Stmt *S) { SubStmts[REMAINDER_BODY] = S; }

  /// A reference to the original loop induction variable, nullptr if the
  /// VarDecl is declared before the loop
  Expr *getLoopIVOrigin() const {
    return cast_if_present<Expr>(SubStmts[LOOP_IV_ORIGIN]);
  }
  void setLoopIVOrigin(Expr *E) {
    SubStmts[LOOP_IV_ORIGIN] = cast_if_present<Stmt>(E);
  }

  /// An expression that sets the loop induction variable to the loop upper
  /// bound
  Expr *getEndLoopIVUpdate() const {
    return cast_if_present<Expr>(SubStmts[LOOP_IV_SET_TO_UB]);
  }
  void setEndLoopIVUpdate(Expr *E) {
    SubStmts[LOOP_IV_SET_TO_UB] = cast_if_present<Stmt>(E);
  }
};

inline raw_ostream &operator<<(raw_ostream &OS,
                               const RippleComputeConstruct &RCC) {
  RCC.print(OS);
  return OS;
}

} // namespace clang
#endif // LLVM_CLANG_AST_STMTRIPPLE_H
