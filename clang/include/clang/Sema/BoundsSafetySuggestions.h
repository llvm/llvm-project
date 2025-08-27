/* TO_UPSTREAM(BoundsSafety) ON */
//===- BoundsSafetySuggestions.h - -fbounds-safety suggestions --*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//  This file defines a collection of analyses that aid adoption of
//  -fbounds-safety annotations.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_SEMA_BOUNDSSAFETYSUGGESTIONS_H
#define LLVM_CLANG_SEMA_BOUNDSSAFETYSUGGESTIONS_H

#include "clang/AST/TypeBase.h"
#include "clang/Basic/Diagnostic.h"
#include "llvm/ADT/ArrayRef.h"
namespace clang {

// Forward-declare to reduce bloat.
class ASTContext;
class Decl;
class Expr;
class FieldDecl;
class FunctionDecl;
class NamedDecl;
class ParmVarDecl;
class Sema;
class Stmt;
class VarDecl;
// An abstract interface to the bounds safety suggestion machine.
class BoundsSafetySuggestionHandler {
public:
  enum class UnsafeOpKind {
    Index,
    Arithmetic,
    Deref,
    MemberAccess,
    Assignment,
    Return,
    CallArg,
    Cast
  };

  enum class AssignmentSourceKind {
    Parameter = 0,
    GlobalVar,
    LocalVar,
    FunctionCallReturnValue,
    ArrayElement,
    StructMember,
    UnionMember,
  };

  enum class WillTrapKind {
    NoTrap,
    Unknown,
    Trap,
    TrapIffPtrNotNull,
    TrapIffPtrNull
  };

  enum class PtrArithOOBKind {
    NEVER_OOB = 0,
    ALWAYS_OOB_BASE_OOB,
    ALWAYS_OOB_CONSTANT_OFFSET,
    OOB_IF_EFFECTIVE_OFFSET_GTE_MINIMUM_OOB_POSITIVE_OFFSET,
    OOB_IF_EFFECTIVE_OFFSET_GTE_MINIMUM_OOB_POSITIVE_OFFSET_OR_LT_ZERO,
    OOB_IF_EFFECTIVE_OFFSET_LT_ZERO,
    UNKNOWN
  };

  struct SingleEntity {
    AssignmentSourceKind Kind;
    const NamedDecl *Entity;
    const Expr *AssignmentExpr;

    // The bounds of the __bidi_indexable will store the bounds of a single
    // `SinglePointeeTy`. Not necessarily the same as the
    // `AssignmentExpr->getType()` due to additional casts.
    const QualType SinglePointeeTy;
  };

  /// Callback for when analysis in UnsafeOperationVisitor detects that an
  /// implicitly __bidi_indexable local variable (that participates in
  /// potentially unsafe pointer arithmetic or indexing) is initialized or
  /// assigned from a an entity that is a __single pointer.
  virtual void handleSingleEntitiesFlowingToIndexableVariableIndexOrPtrArith(
      const llvm::ArrayRef<SingleEntity> Entities,
      const VarDecl *IndexableLocalVar, const Stmt *UnsafeOp, UnsafeOpKind Kind,
      PtrArithOOBKind IsOOB, size_t MinimumPtrArithOOBOffset) = 0;

  /// Callback for when analysis in UnsafeOperationVisitor detects that an
  /// implicitly __bidi_indexable local variable (that participates in
  /// an unsafe buffer operation that accesses the 0th element) is initialized
  /// or assigned from an entity that is a __single pointer.
  virtual void handleSingleEntitiesFlowingToIndexableVariableWithEltZeroOOB(
      const llvm::ArrayRef<SingleEntity> Entities,
      const VarDecl *IndexableLocalVar, const Stmt *UnsafeOp,
      const Expr *Operand, UnsafeOpKind Kind) = 0;

  /// Callback for when analysis in UnsafeOperationVisitor detects that an
  /// implicitly __bidi_indexable local variable (that is
  /// "unsafely cast") is initialized
  /// or assigned from an entity that is a __single pointer.
  ///
  /// The following types of cast are considered unsafe:
  ///
  /// 1. `__bidi_indexable` -> `__single` (CK_BoundsSafetyPointerCast) where
  /// `local` has insufficient bounds to allow access to a single element of the
  /// pointee type.
  ///
  /// 2. Bit cast where the pointee type of the pointer (CK_BitCast) is changed
  /// to a larger type than `local` has the bounds for. I.e. the resulting
  /// pointer is an out-of-bounds pointer.
  virtual void handleSingleEntitiesFlowingToIndexableVariableUnsafelyCasted(
      const llvm::ArrayRef<SingleEntity> Entities,
      const VarDecl *IndexableLocalVar, const Stmt *UnsafeOp, UnsafeOpKind Kind,
      const Expr *Operand) = 0;

  virtual void handleSingleEntitiesFlowingToIndexableDynamicCountConversion(
      const llvm::ArrayRef<SingleEntity> Entities,
      const VarDecl *IndexableLocalVar, const Stmt *UnsafeOp, UnsafeOpKind Kind,
      const Expr *Operand, const QualType DCPT, WillTrapKind WillTrap,
      std::optional<llvm::APInt> ConstantCount, size_t MaxSafeSizeOrCount) = 0;

  // Always provide a virtual destructor!
  virtual ~BoundsSafetySuggestionHandler() = default;
};

void checkBoundsSafetySuggestions(const Decl *D,
                                  BoundsSafetySuggestionHandler &H, Sema &S);

inline const StreamingDiagnostic &
operator<<(const StreamingDiagnostic &PD,
           const BoundsSafetySuggestionHandler::AssignmentSourceKind Kind) {
  PD.AddTaggedVal(static_cast<uint64_t>(Kind), DiagnosticsEngine::ak_uint);
  return PD;
}

} // namespace clang

#endif /* LLVM_CLANG_SEMA_BOUNDSSAFETYSUGGESTIONS_H */
/* TO_UPSTREAM(BoundsSafety) OFF */
