//===-- StorageLocation.h ---------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines classes that represent elements of the local variable store
// and of the heap during dataflow analysis.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_ANALYSIS_FLOWSENSITIVE_STORAGELOCATION_H
#define LLVM_CLANG_ANALYSIS_FLOWSENSITIVE_STORAGELOCATION_H

#include "clang/AST/Decl.h"
#include "clang/AST/Type.h"
#include "llvm/ADT/DenseMap.h"

namespace clang {
namespace dataflow {

/// Base class for elements of the local variable store and of the heap.
///
/// Each storage location holds a value. The mapping from storage locations to
/// values is stored in the environment.
class StorageLocation {
public:
  enum class Kind { Scalar, Aggregate };

  StorageLocation(Kind LocKind, QualType Type) : LocKind(LocKind), Type(Type) {}

  virtual ~StorageLocation() = default;

  Kind getKind() const { return LocKind; }

  QualType getType() const { return Type; }

private:
  Kind LocKind;
  QualType Type;
};

/// A storage location that is not subdivided further for the purposes of
/// abstract interpretation. For example: `int`, `int*`, `int&`.
class ScalarStorageLocation final : public StorageLocation {
public:
  explicit ScalarStorageLocation(QualType Type)
      : StorageLocation(Kind::Scalar, Type) {}

  static bool classof(const StorageLocation *Loc) {
    return Loc->getKind() == Kind::Scalar;
  }
};

/// A storage location which is subdivided into smaller storage locations that
/// can be traced independently by abstract interpretation. For example: a
/// struct with public members.
class AggregateStorageLocation final : public StorageLocation {
public:
  explicit AggregateStorageLocation(QualType Type)
      : AggregateStorageLocation(
            Type, llvm::DenseMap<const ValueDecl *, StorageLocation *>()) {}

  AggregateStorageLocation(
      QualType Type,
      llvm::DenseMap<const ValueDecl *, StorageLocation *> Children)
      : StorageLocation(Kind::Aggregate, Type), Children(std::move(Children)) {}

  static bool classof(const StorageLocation *Loc) {
    return Loc->getKind() == Kind::Aggregate;
  }

  /// Returns the child storage location for `D`.
  StorageLocation &getChild(const ValueDecl &D) const {
    auto It = Children.find(&D);
    assert(It != Children.end());
    return *It->second;
  }

private:
  llvm::DenseMap<const ValueDecl *, StorageLocation *> Children;
};

} // namespace dataflow
} // namespace clang

#endif // LLVM_CLANG_ANALYSIS_FLOWSENSITIVE_STORAGELOCATION_H
