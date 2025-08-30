//===-- tools/extra/clang-reorder-fields/utils/Designator.h -----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains the declarations of the Designator and Designators
/// utility classes.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_REORDER_FIELDS_UTILS_DESIGNATOR_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_REORDER_FIELDS_UTILS_DESIGNATOR_H

#include "clang/AST/Decl.h"
#include "clang/AST/Expr.h"
#include "clang/AST/Type.h"

namespace clang {
namespace reorder_fields {

/// Represents a part of a designation in a C99/C++20 designated initializer. It
/// is a tagged union of different kinds of designators: struct, array and array
/// range. Holds enough information to be able to advance to the next field and
/// to know when all fields have been iterated through.
class Designator {
public:
  enum Kind { STRUCT, ARRAY, ARRAY_RANGE };

  Designator(const QualType Type, RecordDecl::field_iterator Field,
             const RecordDecl *RD)
      : Tag(STRUCT), Type(Type), StructIt({Field, RD}) {}

  Designator(const QualType Type, uint64_t Idx, uint64_t Size)
      : Tag(ARRAY), Type(Type), ArrayIt({Idx, Size}) {}

  Designator(const QualType Type, uint64_t Start, uint64_t End, uint64_t Size)
      : Tag(ARRAY_RANGE), Type(Type), ArrayRangeIt({Start, End, Size}) {}

  /// Moves the iterator to the next element.
  void advanceToNextField();

  /// Checks if the iterator has iterated through all elements.
  bool isFinished();

  Kind getTag() const { return Tag; }
  QualType getType() const { return Type; }

  const RecordDecl::field_iterator getStructIter() const {
    assert(Tag == STRUCT && "Must be a field designator");
    return StructIt.Field;
  }

  const RecordDecl *getStructDecl() const {
    assert(Tag == STRUCT && "Must be a field designator");
    return StructIt.Record;
  }

  uint64_t getArrayIndex() const {
    assert(Tag == ARRAY && "Must be an array designator");
    return ArrayIt.Index;
  }

  uint64_t getArrayRangeStart() const {
    assert(Tag == ARRAY_RANGE && "Must be an array range designator");
    return ArrayRangeIt.Start;
  }

  uint64_t getArrayRangeEnd() const {
    assert(Tag == ARRAY_RANGE && "Must be an array range designator");
    return ArrayRangeIt.End;
  }

  uint64_t getArraySize() const {
    assert((Tag == ARRAY || Tag == ARRAY_RANGE) &&
           "Must be an array or range designator");
    if (Tag == ARRAY)
      return ArrayIt.Size;
    return ArrayRangeIt.Size;
  }

private:
  /// Type of the designator.
  Kind Tag;

  /// Type of the designated entry. For arrays this is the type of the element.
  QualType Type;

  /// Field designator has the iterator to the field and the record the field
  /// is declared in.
  struct StructIter {
    RecordDecl::field_iterator Field;
    const RecordDecl *Record;
  };

  /// Array designator has an index and size of the array.
  struct ArrayIter {
    uint64_t Index;
    uint64_t Size;
  };

  /// Array range designator has a start and end index and size of the array.
  struct ArrayRangeIter {
    uint64_t Start;
    uint64_t End;
    uint64_t Size;
  };

  union {
    StructIter StructIt;
    ArrayIter ArrayIt;
    ArrayRangeIter ArrayRangeIt;
  };
};

/// List of designators.
class Designators {
public:
  /// Initialize to the first member of the struct/array. Enters implicit
  /// initializer lists until a type that matches Init is found.
  Designators(const Expr *Init, const InitListExpr *ILE,
              const ASTContext *Context);

  /// Initialize to the designators of the given expression.
  Designators(const DesignatedInitExpr *DIE, const InitListExpr *ILE,
              const ASTContext *Context);

  /// Return whether this designator list is valid.
  bool isValid() const { return !DesignatorList.empty(); }

  /// Moves the designators to the next initializer in the struct/array. If the
  /// type of next initializer doesn't match the expected type then there are
  /// omitted braces and we add new designators to reflect that.
  bool advanceToNextField(const Expr *Init);

  /// Gets a string representation from a list of designators. This string will
  /// be inserted before an initializer expression to make it designated.
  std::string toString() const;

  size_t size() const { return DesignatorList.size(); }

  SmallVector<Designator>::const_iterator begin() const {
    return DesignatorList.begin();
  }
  SmallVector<Designator>::const_iterator end() const {
    return DesignatorList.end();
  }

private:
  /// Enters any implicit initializer lists until a type that matches the given
  /// expression is found.
  bool enterImplicitInitLists(const Expr *Init);

  const InitListExpr *ILE;
  const ASTContext *Context;
  SmallVector<Designator, 1> DesignatorList;
};

} // namespace reorder_fields
} // namespace clang

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_REORDER_FIELDS_UTILS_DESIGNATOR_H
