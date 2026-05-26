//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains implementation details, such as storage structures, of
// CIR dialect types.
//
//===----------------------------------------------------------------------===//
#ifndef CIR_DIALECT_IR_CIRTYPESDETAILS_H
#define CIR_DIALECT_IR_CIRTYPESDETAILS_H

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/Support/LogicalResult.h"
#include "clang/CIR/Dialect/IR/CIRTypes.h"
#include "llvm/ADT/Hashing.h"

namespace cir {
namespace detail {

//===----------------------------------------------------------------------===//
// CIR RecordTypeStorage
//===----------------------------------------------------------------------===//

/// Type storage for CIR record types.
struct RecordTypeStorage : public mlir::TypeStorage {
  struct KeyTy {
    llvm::ArrayRef<mlir::Type> members;
    mlir::StringAttr name;
    bool incomplete;
    bool packed;
    bool padded;
    bool is_class;
    mlir::Type padding;

    KeyTy(llvm::ArrayRef<mlir::Type> members, mlir::StringAttr name,
          bool incomplete, bool packed, bool padded, bool is_class,
          mlir::Type padding)
        : members(members), name(name), incomplete(incomplete), packed(packed),
          padded(padded), is_class(is_class), padding(padding) {}
  };

  llvm::ArrayRef<mlir::Type> members;
  mlir::StringAttr name;
  bool incomplete;
  bool packed;
  bool padded;
  bool is_class;
  mlir::Type padding;

  RecordTypeStorage(llvm::ArrayRef<mlir::Type> members, mlir::StringAttr name,
                    bool incomplete, bool packed, bool padded, bool is_class,
                    mlir::Type padding)
      : members(members), name(name), incomplete(incomplete), packed(packed),
        padded(padded), is_class(is_class), padding(padding) {
    assert((name || !incomplete) && "Incomplete records must have a name");
  }

  KeyTy getAsKey() const {
    return KeyTy(members, name, incomplete, packed, padded, is_class, padding);
  }

  bool operator==(const KeyTy &key) const {
    if (name)
      return (name == key.name) && (is_class == key.is_class);
    return std::tie(members, name, incomplete, packed, padded, is_class,
                    padding) == std::tie(key.members, key.name, key.incomplete,
                                         key.packed, key.padded, key.is_class,
                                         key.padding);
  }

  static llvm::hash_code hashKey(const KeyTy &key) {
    if (key.name)
      return llvm::hash_combine(key.name, key.is_class);
    return llvm::hash_combine(key.members, key.incomplete, key.packed,
                              key.padded, key.is_class, key.padding);
  }

  static RecordTypeStorage *construct(mlir::TypeStorageAllocator &allocator,
                                      const KeyTy &key) {
    return new (allocator.allocate<RecordTypeStorage>()) RecordTypeStorage(
        allocator.copyInto(key.members), key.name, key.incomplete, key.packed,
        key.padded, key.is_class, key.padding);
  }

  /// Mutates the members and attributes an identified record.
  ///
  /// Once a record is mutated, it is marked as complete, preventing further
  /// mutations.  Anonymous records are always complete and cannot be mutated.
  /// This method does not fail if a mutation of a complete record does not
  /// change the record.
  llvm::LogicalResult mutate(mlir::TypeStorageAllocator &allocator,
                             llvm::ArrayRef<mlir::Type> members, bool packed,
                             bool padded, mlir::Type padding) {
    // Anonymous records cannot mutate.
    if (!name)
      return llvm::failure();

    // Mutation of complete records are allowed if they change nothing.
    if (!incomplete)
      return mlir::success(
          (this->members == members) && (this->packed == packed) &&
          (this->padded == padded) && (this->padding == padding));

    // Mutate incomplete record.
    this->members = allocator.copyInto(members);
    this->packed = packed;
    this->padded = padded;
    this->padding = padding;

    incomplete = false;
    return llvm::success();
  }
};

} // namespace detail
} // namespace cir

#endif // CIR_DIALECT_IR_CIRTYPESDETAILS_H
