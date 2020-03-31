//===--- ExprOpenMP.h - Classes for representing expressions ----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//  This file defines the Expr interface and subclasses.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_AST_EXPROPENMP_H
#define LLVM_CLANG_AST_EXPROPENMP_H

#include "clang/AST/ComputeDependence.h"
#include "clang/AST/Expr.h"

namespace clang {
/// OpenMP 4.0 [2.4, Array Sections].
/// To specify an array section in an OpenMP construct, array subscript
/// expressions are extended with the following syntax:
/// \code
/// [ lower-bound : length ]
/// [ lower-bound : ]
/// [ : length ]
/// [ : ]
/// \endcode
/// The array section must be a subset of the original array.
/// Array sections are allowed on multidimensional arrays. Base language array
/// subscript expressions can be used to specify length-one dimensions of
/// multidimensional array sections.
/// The lower-bound and length are integral type expressions. When evaluated
/// they represent a set of integer values as follows:
/// \code
/// { lower-bound, lower-bound + 1, lower-bound + 2,... , lower-bound + length -
/// 1 }
/// \endcode
/// The lower-bound and length must evaluate to non-negative integers.
/// When the size of the array dimension is not known, the length must be
/// specified explicitly.
/// When the length is absent, it defaults to the size of the array dimension
/// minus the lower-bound.
/// When the lower-bound is absent it defaults to 0.
class OMPArraySectionExpr : public Expr {
  enum { BASE, LOWER_BOUND, LENGTH, END_EXPR };
  Stmt *SubExprs[END_EXPR];
  SourceLocation ColonLoc;
  SourceLocation RBracketLoc;

public:
  OMPArraySectionExpr(Expr *Base, Expr *LowerBound, Expr *Length, QualType Type,
                      ExprValueKind VK, ExprObjectKind OK,
                      SourceLocation ColonLoc, SourceLocation RBracketLoc)
      : Expr(OMPArraySectionExprClass, Type, VK, OK), ColonLoc(ColonLoc),
        RBracketLoc(RBracketLoc) {
    SubExprs[BASE] = Base;
    SubExprs[LOWER_BOUND] = LowerBound;
    SubExprs[LENGTH] = Length;
    setDependence(computeDependence(this));
  }

  /// Create an empty array section expression.
  explicit OMPArraySectionExpr(EmptyShell Shell)
      : Expr(OMPArraySectionExprClass, Shell) {}

  /// An array section can be written only as Base[LowerBound:Length].

  /// Get base of the array section.
  Expr *getBase() { return cast<Expr>(SubExprs[BASE]); }
  const Expr *getBase() const { return cast<Expr>(SubExprs[BASE]); }
  /// Set base of the array section.
  void setBase(Expr *E) { SubExprs[BASE] = E; }

  /// Return original type of the base expression for array section.
  static QualType getBaseOriginalType(const Expr *Base);

  /// Get lower bound of array section.
  Expr *getLowerBound() { return cast_or_null<Expr>(SubExprs[LOWER_BOUND]); }
  const Expr *getLowerBound() const {
    return cast_or_null<Expr>(SubExprs[LOWER_BOUND]);
  }
  /// Set lower bound of the array section.
  void setLowerBound(Expr *E) { SubExprs[LOWER_BOUND] = E; }

  /// Get length of array section.
  Expr *getLength() { return cast_or_null<Expr>(SubExprs[LENGTH]); }
  const Expr *getLength() const { return cast_or_null<Expr>(SubExprs[LENGTH]); }
  /// Set length of the array section.
  void setLength(Expr *E) { SubExprs[LENGTH] = E; }

  SourceLocation getBeginLoc() const LLVM_READONLY {
    return getBase()->getBeginLoc();
  }
  SourceLocation getEndLoc() const LLVM_READONLY { return RBracketLoc; }

  SourceLocation getColonLoc() const { return ColonLoc; }
  void setColonLoc(SourceLocation L) { ColonLoc = L; }

  SourceLocation getRBracketLoc() const { return RBracketLoc; }
  void setRBracketLoc(SourceLocation L) { RBracketLoc = L; }

  SourceLocation getExprLoc() const LLVM_READONLY {
    return getBase()->getExprLoc();
  }

  static bool classof(const Stmt *T) {
    return T->getStmtClass() == OMPArraySectionExprClass;
  }

  child_range children() {
    return child_range(&SubExprs[BASE], &SubExprs[END_EXPR]);
  }

  const_child_range children() const {
    return const_child_range(&SubExprs[BASE], &SubExprs[END_EXPR]);
  }
};

/// An explicit cast in C or a C-style cast in C++, which uses the syntax
/// ([s1][s2]...[sn])expr. For example: @c ([3][3])f.
class OMPArrayShapingExpr final
    : public Expr,
      private llvm::TrailingObjects<OMPArrayShapingExpr, Expr *, SourceRange> {
  friend TrailingObjects;
  friend class ASTStmtReader;
  friend class ASTStmtWriter;
  /// Base node.
  SourceLocation LPLoc; /// The location of the left paren
  SourceLocation RPLoc; /// The location of the right paren
  unsigned NumDims = 0; /// Number of dimensions in the shaping expression.

  /// Construct full expression.
  OMPArrayShapingExpr(QualType ExprTy, Expr *Op, SourceLocation L,
                      SourceLocation R, ArrayRef<Expr *> Dims);

  /// Construct an empty expression.
  explicit OMPArrayShapingExpr(EmptyShell Shell, unsigned NumDims)
      : Expr(OMPArrayShapingExprClass, Shell), NumDims(NumDims) {}

  /// Sets the dimensions for the array shaping.
  void setDimensions(ArrayRef<Expr *> Dims);

  /// Sets the base expression for array shaping operation.
  void setBase(Expr *Op) { getTrailingObjects<Expr *>()[NumDims] = Op; }

  /// Sets source ranges for the brackets in the array shaping operation.
  void setBracketsRanges(ArrayRef<SourceRange> BR);

  unsigned numTrailingObjects(OverloadToken<Expr *>) const {
    // Add an extra one for the base expression.
    return NumDims + 1;
  }

  unsigned numTrailingObjects(OverloadToken<SourceRange>) const {
    return NumDims;
  }

public:
  static OMPArrayShapingExpr *Create(const ASTContext &Context, QualType T,
                                     Expr *Op, SourceLocation L,
                                     SourceLocation R, ArrayRef<Expr *> Dims,
                                     ArrayRef<SourceRange> BracketRanges);

  static OMPArrayShapingExpr *CreateEmpty(const ASTContext &Context,
                                          unsigned NumDims);

  SourceLocation getLParenLoc() const { return LPLoc; }
  void setLParenLoc(SourceLocation L) { LPLoc = L; }

  SourceLocation getRParenLoc() const { return RPLoc; }
  void setRParenLoc(SourceLocation L) { RPLoc = L; }

  SourceLocation getBeginLoc() const LLVM_READONLY { return LPLoc; }
  SourceLocation getEndLoc() const LLVM_READONLY {
    return getBase()->getEndLoc();
  }

  /// Fetches the dimensions for array shaping expression.
  ArrayRef<Expr *> getDimensions() const {
    return llvm::makeArrayRef(getTrailingObjects<Expr *>(), NumDims);
  }

  /// Fetches source ranges for the brackets os the array shaping expression.
  ArrayRef<SourceRange> getBracketsRanges() const {
    return llvm::makeArrayRef(getTrailingObjects<SourceRange>(), NumDims);
  }

  /// Fetches base expression of array shaping expression.
  Expr *getBase() { return getTrailingObjects<Expr *>()[NumDims]; }
  const Expr *getBase() const { return getTrailingObjects<Expr *>()[NumDims]; }

  static bool classof(const Stmt *T) {
    return T->getStmtClass() == OMPArrayShapingExprClass;
  }

  // Iterators
  child_range children() {
    Stmt **Begin = reinterpret_cast<Stmt **>(getTrailingObjects<Expr *>());
    return child_range(Begin, Begin + NumDims + 1);
  }
  const_child_range children() const {
    Stmt *const *Begin =
        reinterpret_cast<Stmt *const *>(getTrailingObjects<Expr *>());
    return const_child_range(Begin, Begin + NumDims + 1);
  }
};
} // end namespace clang

#endif
