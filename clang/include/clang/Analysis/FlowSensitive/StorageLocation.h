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
#include "llvm/Support/Debug.h"
#include <cassert>

#define DEBUG_TYPE "dataflow"

namespace clang {
namespace dataflow {

/// Base class for elements of the local variable store and of the heap.
///
/// Each storage location holds a value. The mapping from storage locations to
/// values is stored in the environment.
class StorageLocation {
public:
  enum class Kind { Scalar, Aggregate };

  StorageLocation(Kind LocKind, QualType Type) : LocKind(LocKind), Type(Type) {
    assert(Type.isNull() || !Type->isReferenceType());
  }

  // Non-copyable because addresses of storage locations are used as their
  // identities throughout framework and user code. The framework is responsible
  // for construction and destruction of storage locations.
  StorageLocation(const StorageLocation &) = delete;
  StorageLocation &operator=(const StorageLocation &) = delete;

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
/// struct with public members. The child map is flat, so when used for a struct
/// or class type, all accessible members of base struct and class types are
/// directly accesible as children of this location.
///
/// The storage location for a field of reference type may be null. This
/// typically occurs in one of two situations:
/// - The record has not been fully initialized.
/// - The maximum depth for modelling a self-referential data structure has been
///   reached.
/// Storage locations for fields of all other types must be non-null.
///
/// FIXME: Currently, the storage location of unions is modelled the same way as
/// that of structs or classes. Eventually, we need to change this modelling so
/// that all of the members of a given union have the same storage location.
class AggregateStorageLocation final : public StorageLocation {
public:
  using FieldToLoc = llvm::DenseMap<const ValueDecl *, StorageLocation *>;

  explicit AggregateStorageLocation(QualType Type)
      : AggregateStorageLocation(Type, FieldToLoc()) {}

  AggregateStorageLocation(QualType Type, FieldToLoc TheChildren)
      : StorageLocation(Kind::Aggregate, Type),
        Children(std::move(TheChildren)) {
    assert(!Type.isNull());
    assert(Type->isRecordType());
    assert([this] {
      for (auto [Field, Loc] : Children) {
        if (!Field->getType()->isReferenceType() && Loc == nullptr)
          return false;
      }
      return true;
    }());
  }

  static bool classof(const StorageLocation *Loc) {
    return Loc->getKind() == Kind::Aggregate;
  }

  /// Returns the child storage location for `D`.
  ///
  /// May return null if `D` has reference type; guaranteed to return non-null
  /// in all other cases.
  ///
  /// Note that it is an error to call this with a field that does not exist.
  /// The function does not return null in this case.
  StorageLocation *getChild(const ValueDecl &D) const {
    auto It = Children.find(&D);
    LLVM_DEBUG({
      if (It == Children.end()) {
        llvm::dbgs() << "Couldn't find child " << D.getNameAsString()
                     << " on StorageLocation " << this << " of type "
                     << getType() << "\n";
        llvm::dbgs() << "Existing children:\n";
        for ([[maybe_unused]] auto [Field, Loc] : Children) {
          llvm::dbgs() << Field->getNameAsString() << "\n";
        }
      }
    });
    assert(It != Children.end());
    return It->second;
  }

  /// Changes the child storage location for a field `D` of reference type.
  /// All other fields cannot change their storage location and always retain
  /// the storage location passed to the `AggregateStorageLocation` constructor.
  ///
  /// Requirements:
  ///
  ///  `D` must have reference type.
  void setChild(const ValueDecl &D, StorageLocation *Loc) {
    assert(D.getType()->isReferenceType());
    Children[&D] = Loc;
  }

  llvm::iterator_range<FieldToLoc::const_iterator> children() const {
    return {Children.begin(), Children.end()};
  }

private:
  FieldToLoc Children;
};

} // namespace dataflow
} // namespace clang

#undef DEBUG_TYPE

#endif // LLVM_CLANG_ANALYSIS_FLOWSENSITIVE_STORAGELOCATION_H
