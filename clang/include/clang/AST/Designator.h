//===--- Designator.h - Initialization Designator ---------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines interfaces used to represent designators (i.e. C99
// designated initializers) during parsing and sema.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_AST_DESIGNATOR_H
#define LLVM_CLANG_AST_DESIGNATOR_H

#include "clang/Basic/SourceLocation.h"
#include "llvm/ADT/SmallVector.h"

namespace clang {

class Expr;
class FieldDecl;
class IdentifierInfo;
class Sema;

/// Designator - A designator in a C99 designated initializer.
///
/// This class is a discriminated union which holds the various
/// different sorts of designators possible. A Designation is an array of
/// these.  An example of a designator are things like this:
///
///      [8] .field [47]        // C99 designation: 3 designators
///      [8 ... 47]  field:     // GNU extensions: 2 designators
///
/// These occur in initializers, e.g.:
///
///      int a[10] = {2, 4, [8]=9, 10};
///
class Designator {
  enum DesignatorKind {
    FieldDesignator,
    ArrayDesignatorExpr,
    ArrayDesignatorInt,
    ArrayRangeDesignatorExpr,
    ArrayRangeDesignatorInt,
  };

  DesignatorKind Kind;

  /// A field designator, e.g., ".x = 42".
  class FieldDesignatorInfo {
    /// Refers to the field that is being initialized. The low bit of this
    /// field determines whether this is actually a pointer to an
    /// IdentifierInfo (if 1) or a FieldDecl (if 0). When initially
    /// constructed, a field designator will store an IdentifierInfo*. After
    /// semantic analysis has resolved that name, the field designator will
    /// instead store a FieldDecl*.
    uintptr_t NameOrField;

  public:
    /// The location of the '.' in the designated initializer.
    SourceLocation DotLoc;

    /// The location of the field name in the designated initializer.
    SourceLocation NameLoc;

    FieldDesignatorInfo(const IdentifierInfo *II, SourceLocation DotLoc,
                        SourceLocation NameLoc)
        : NameOrField(reinterpret_cast<uintptr_t>(II) | 0x01), DotLoc(DotLoc),
          NameLoc(NameLoc) {}

    FieldDecl *getFieldDecl() const {
      if (NameOrField & 0x01)
        return nullptr;
      return reinterpret_cast<FieldDecl *>(NameOrField);
    }
    const IdentifierInfo *getIdentifierInfo() const {
      if (NameOrField & 0x01)
        return reinterpret_cast<const IdentifierInfo *>(NameOrField & ~0x1);
      return nullptr;
    }

    void set(FieldDecl *FD) { NameOrField = reinterpret_cast<uintptr_t>(FD); }
  };

  /// An array designator, e.g., "[42] = 0" and "[42 ... 50] = 1".
  template <typename Ty> struct ArrayDesignatorInfo {
    /// Location of the first and last index expression within the designated
    /// initializer expression's list of subexpressions.
    Ty Start;
    Ty End;

    /// The location of the '[' starting the array designator.
    SourceLocation LBracketLoc;

    /// The location of the ellipsis separating the start and end indices.
    /// Only valid for GNU array-range designators.
    SourceLocation EllipsisLoc;

    /// The location of the ']' terminating the array designator.
    SourceLocation RBracketLoc;

    ArrayDesignatorInfo(Ty Start, Ty End, SourceLocation LBracketLoc,
                        SourceLocation EllipsisLoc, SourceLocation RBracketLoc)
        : Start(Start), End(End), LBracketLoc(LBracketLoc),
          EllipsisLoc(EllipsisLoc), RBracketLoc(RBracketLoc) {}

    Ty getStart() const { return Start; }
    Ty getEnd() const { return End; }
  };

  union {
    FieldDesignatorInfo FieldInfo;
    ArrayDesignatorInfo<Expr *> ArrayInfoExpr;
    ArrayDesignatorInfo<unsigned> ArrayInfoInt;
  };

  Designator(DesignatorKind Kind) : Kind(Kind) {}

public:
  Designator() {}

  bool isFieldDesignator() const { return Kind == FieldDesignator; }
  bool isArrayDesignator() const {
    return Kind == ArrayDesignatorExpr || Kind == ArrayDesignatorInt;
  }
  bool isArrayRangeDesignator() const {
    return Kind == ArrayRangeDesignatorExpr || Kind == ArrayRangeDesignatorInt;
  }

  /// FieldDesignatorInfo:
  static Designator CreateFieldDesignator(const IdentifierInfo *FieldName,
                                          SourceLocation DotLoc,
                                          SourceLocation NameLoc) {
    Designator D(FieldDesignator);
    new (&D.FieldInfo) FieldDesignatorInfo(FieldName, DotLoc, NameLoc);
    return D;
  }

  const IdentifierInfo *getFieldName() const;

  FieldDecl *getField() const {
    assert(isFieldDesignator() && "Invalid accessor");
    return FieldInfo.getFieldDecl();
  }

  void setField(FieldDecl *FD) {
    assert(isFieldDesignator() && "Invalid accessor");
    FieldInfo.set(FD);
  }

  SourceLocation getDotLoc() const {
    assert(isFieldDesignator() && "Invalid accessor");
    return FieldInfo.DotLoc;
  }

  SourceLocation getFieldLoc() const {
    assert(isFieldDesignator() && "Invalid accessor");
    return FieldInfo.NameLoc;
  }

  /// ArrayDesignatorInfo:
  static Designator
  CreateArrayDesignator(Expr *Start, SourceLocation LBracketLoc,
                        SourceLocation RBracketLoc = SourceLocation()) {
    Designator D(ArrayDesignatorExpr);
    new (&D.ArrayInfoExpr) ArrayDesignatorInfo<Expr *>(
        Start, nullptr, LBracketLoc, SourceLocation(), RBracketLoc);
    return D;
  }
  static Designator
  CreateArrayDesignator(unsigned Start, SourceLocation LBracketLoc,
                        SourceLocation RBracketLoc = SourceLocation()) {
    Designator D(ArrayDesignatorInt);
    new (&D.ArrayInfoInt) ArrayDesignatorInfo<unsigned>(
        Start, 0, LBracketLoc, SourceLocation(), RBracketLoc);
    return D;
  }

  template <typename Ty = Expr *> Ty getArrayIndex() const {
    assert(isArrayDesignator() && "Invalid accessor");
    return ArrayInfoExpr.getStart();
  }

  /// ArrayRangeDesignatorInfo:
  static Designator
  CreateArrayRangeDesignator(Expr *Start, Expr *End, SourceLocation LBracketLoc,
                             SourceLocation EllipsisLoc,
                             SourceLocation RBracketLoc = SourceLocation()) {
    Designator D(ArrayRangeDesignatorExpr);
    new (&D.ArrayInfoExpr) ArrayDesignatorInfo<Expr *>(
        Start, End, LBracketLoc, EllipsisLoc, RBracketLoc);
    return D;
  }
  static Designator
  CreateArrayRangeDesignator(unsigned Index, SourceLocation LBracketLoc,
                             SourceLocation EllipsisLoc,
                             SourceLocation RBracketLoc = SourceLocation()) {
    Designator D(ArrayRangeDesignatorInt);
    new (&D.ArrayInfoInt) ArrayDesignatorInfo<unsigned>(
        Index, 0, LBracketLoc, EllipsisLoc, RBracketLoc);
    return D;
  }

  Expr *getArrayRangeStart() const {
    assert(isArrayRangeDesignator() && "Invalid accessor");
    return ArrayInfoExpr.getStart();
  }

  Expr *getArrayRangeEnd() const {
    assert(isArrayRangeDesignator() && "Invalid accessor");
    return ArrayInfoExpr.getEnd();
  }

  SourceLocation getLBracketLoc() const {
    switch (Kind) {
    default:
      break;
    case ArrayDesignatorExpr:
    case ArrayRangeDesignatorExpr:
      return ArrayInfoExpr.LBracketLoc;
    case ArrayDesignatorInt:
    case ArrayRangeDesignatorInt:
      return ArrayInfoInt.LBracketLoc;
    }

    assert(false && "Invalid accessor");
    return SourceLocation();
  }

  SourceLocation getEllipsisLoc() const {
    switch (Kind) {
    default:
      break;
    case ArrayRangeDesignatorExpr:
      return ArrayInfoExpr.EllipsisLoc;
    case ArrayRangeDesignatorInt:
      return ArrayInfoInt.EllipsisLoc;
    }

    assert(false && "Invalid accessor");
    return SourceLocation();
  }

  SourceLocation getRBracketLoc() const {
    switch (Kind) {
    default:
      break;
    case ArrayDesignatorExpr:
    case ArrayRangeDesignatorExpr:
      return ArrayInfoExpr.RBracketLoc;
    case ArrayDesignatorInt:
    case ArrayRangeDesignatorInt:
      return ArrayInfoInt.RBracketLoc;
    }

    assert(false && "Invalid accessor");
    return SourceLocation();
  }

  void setRBracketLoc(SourceLocation RBracketLoc) {
    switch (Kind) {
    default:
      assert(false && "Invalid accessor");
      break;
    case ArrayDesignatorExpr:
    case ArrayRangeDesignatorExpr:
      ArrayInfoExpr.RBracketLoc = RBracketLoc;
      break;
    case ArrayDesignatorInt:
    case ArrayRangeDesignatorInt:
      ArrayInfoInt.RBracketLoc = RBracketLoc;
      break;
    }
  }

  unsigned getFirstExprIndex() const {
    if (Kind == ArrayDesignatorInt || Kind == ArrayRangeDesignatorInt)
      return ArrayInfoInt.getStart();

    assert(false && "Invalid accessor");
    return 0;
  }

  /// Source location accessors.
  SourceLocation getBeginLoc() const LLVM_READONLY {
    if (isFieldDesignator())
      return getDotLoc().isInvalid() ? getFieldLoc() : getDotLoc();
    return getLBracketLoc();
  }
  SourceLocation getEndLoc() const LLVM_READONLY {
    return isFieldDesignator() ? getFieldLoc() : getRBracketLoc();
  }
  SourceRange getSourceRange() const LLVM_READONLY {
    return SourceRange(getBeginLoc(), getEndLoc());
  }
};

/// Designation - Represent a full designation, which is a sequence of
/// designators.  This class is mostly a helper for InitListDesignations.
class Designation {
  /// Designators - The actual designators for this initializer.
  SmallVector<Designator, 2> Designators;

public:
  /// AddDesignator - Add a designator to the end of this list.
  void AddDesignator(Designator D) { Designators.push_back(D); }

  bool empty() const { return Designators.empty(); }

  unsigned getNumDesignators() const { return Designators.size(); }

  const Designator &getDesignator(unsigned Idx) const {
    assert(Idx < getNumDesignators());
    return Designators[Idx];
  }
  Designator &getDesignator(unsigned Idx) {
    assert(Idx < getNumDesignators());
    return Designators[Idx];
  }
};

} // end namespace clang

#endif
