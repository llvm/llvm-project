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
// CIR StructTypeStorage
//===----------------------------------------------------------------------===//

/// Type storage for CIR struct/class types.
struct StructTypeStorage : public mlir::TypeStorage {
  struct KeyTy {
    llvm::ArrayRef<mlir::Type> members;
    mlir::StringAttr name;
    bool incomplete;
    bool packed;
    bool padded;
    bool is_class;

    KeyTy(llvm::ArrayRef<mlir::Type> members, mlir::StringAttr name,
          bool incomplete, bool packed, bool padded, bool is_class)
        : members(members), name(name), incomplete(incomplete), packed(packed),
          padded(padded), is_class(is_class) {}
  };

  llvm::ArrayRef<mlir::Type> members;
  mlir::StringAttr name;
  bool incomplete;
  bool packed;
  bool padded;
  bool is_class;

  StructTypeStorage(llvm::ArrayRef<mlir::Type> members, mlir::StringAttr name,
                    bool incomplete, bool packed, bool padded, bool is_class)
      : members(members), name(name), incomplete(incomplete), packed(packed),
        padded(padded), is_class(is_class) {
    assert((name || !incomplete) && "Incomplete records must have a name");
  }

  KeyTy getAsKey() const {
    return KeyTy(members, name, incomplete, packed, padded, is_class);
  }

  bool operator==(const KeyTy &key) const {
    if (name)
      return (name == key.name) && (is_class == key.is_class);
    return std::tie(members, name, incomplete, packed, padded, is_class) ==
           std::tie(key.members, key.name, key.incomplete, key.packed,
                    key.padded, key.is_class);
  }

  static llvm::hash_code hashKey(const KeyTy &key) {
    if (key.name)
      return llvm::hash_combine(key.name, key.is_class);
    return llvm::hash_combine(key.members, key.incomplete, key.packed,
                              key.padded, key.is_class);
  }

  static StructTypeStorage *construct(mlir::TypeStorageAllocator &allocator,
                                      const KeyTy &key) {
    return new (allocator.allocate<StructTypeStorage>())
        StructTypeStorage(allocator.copyInto(key.members), key.name,
                          key.incomplete, key.packed, key.padded, key.is_class);
  }

  /// Mutates the members and attributes of an identified struct/class.
  llvm::LogicalResult mutate(mlir::TypeStorageAllocator &allocator,
                             llvm::ArrayRef<mlir::Type> members, bool packed,
                             bool padded) {
    if (!name)
      return llvm::failure();

    if (!incomplete)
      return mlir::success((this->members == members) &&
                           (this->packed == packed) &&
                           (this->padded == padded));

    this->members = allocator.copyInto(members);
    this->packed = packed;
    this->padded = padded;
    incomplete = false;
    return llvm::success();
  }
};

//===----------------------------------------------------------------------===//
// CIR UnionTypeStorage
//===----------------------------------------------------------------------===//

/// Type storage for CIR union types.
struct UnionTypeStorage : public mlir::TypeStorage {
  struct KeyTy {
    llvm::ArrayRef<mlir::Type> members;
    mlir::StringAttr name;
    bool incomplete;
    bool packed;
    mlir::Type padding;

    KeyTy(llvm::ArrayRef<mlir::Type> members, mlir::StringAttr name,
          bool incomplete, bool packed, mlir::Type padding)
        : members(members), name(name), incomplete(incomplete), packed(packed),
          padding(padding) {}
  };

  llvm::ArrayRef<mlir::Type> members;
  mlir::StringAttr name;
  bool incomplete;
  bool packed;
  mlir::Type padding;

  UnionTypeStorage(llvm::ArrayRef<mlir::Type> members, mlir::StringAttr name,
                   bool incomplete, bool packed, mlir::Type padding)
      : members(members), name(name), incomplete(incomplete), packed(packed),
        padding(padding) {
    assert((name || !incomplete) && "Incomplete records must have a name");
  }

  KeyTy getAsKey() const {
    return KeyTy(members, name, incomplete, packed, padding);
  }

  bool operator==(const KeyTy &key) const {
    if (name)
      return name == key.name;
    return std::tie(members, name, incomplete, packed, padding) ==
           std::tie(key.members, key.name, key.incomplete, key.packed,
                    key.padding);
  }

  static llvm::hash_code hashKey(const KeyTy &key) {
    if (key.name)
      return llvm::hash_combine(key.name);
    return llvm::hash_combine(key.members, key.incomplete, key.packed,
                              key.padding);
  }

  static UnionTypeStorage *construct(mlir::TypeStorageAllocator &allocator,
                                     const KeyTy &key) {
    return new (allocator.allocate<UnionTypeStorage>())
        UnionTypeStorage(allocator.copyInto(key.members), key.name,
                         key.incomplete, key.packed, key.padding);
  }

  /// Mutates the members and attributes of an identified union.
  llvm::LogicalResult mutate(mlir::TypeStorageAllocator &allocator,
                             llvm::ArrayRef<mlir::Type> members, bool packed,
                             mlir::Type padding) {
    if (!name)
      return llvm::failure();

    if (!incomplete)
      return mlir::success((this->members == members) &&
                           (this->packed == packed) &&
                           (this->padding == padding));

    this->members = allocator.copyInto(members);
    this->packed = packed;
    this->padding = padding;
    incomplete = false;
    return llvm::success();
  }
};

} // namespace detail
} // namespace cir

#endif // CIR_DIALECT_IR_CIRTYPESDETAILS_H
