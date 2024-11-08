//===- CIRTypesDetails.h - Details of CIR dialect types ---------*- C++ -*-===//
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

/// Type storage for CIR record types.
struct StructTypeStorage : public mlir::TypeStorage {
  struct KeyTy {
    llvm::ArrayRef<mlir::Type> members;
    mlir::StringAttr name;
    bool incomplete;
    bool packed;
    StructType::RecordKind kind;
    ASTRecordDeclInterface ast;

    KeyTy(llvm::ArrayRef<mlir::Type> members, mlir::StringAttr name,
          bool incomplete, bool packed, StructType::RecordKind kind,
          ASTRecordDeclInterface ast)
        : members(members), name(name), incomplete(incomplete), packed(packed),
          kind(kind), ast(ast) {}
  };

  llvm::ArrayRef<mlir::Type> members;
  mlir::StringAttr name;
  bool incomplete;
  bool packed;
  StructType::RecordKind kind;
  ASTRecordDeclInterface ast;

  StructTypeStorage(llvm::ArrayRef<mlir::Type> members, mlir::StringAttr name,
                    bool incomplete, bool packed, StructType::RecordKind kind,
                    ASTRecordDeclInterface ast)
      : members(members), name(name), incomplete(incomplete), packed(packed),
        kind(kind), ast(ast) {}

  KeyTy getAsKey() const {
    return KeyTy(members, name, incomplete, packed, kind, ast);
  }

  bool operator==(const KeyTy &key) const {
    if (name)
      return (name == key.name) && (kind == key.kind);
    return (members == key.members) && (name == key.name) &&
           (incomplete == key.incomplete) && (packed == key.packed) &&
           (kind == key.kind) && (ast == key.ast);
  }

  static llvm::hash_code hashKey(const KeyTy &key) {
    if (key.name)
      return llvm::hash_combine(key.name, key.kind);
    return llvm::hash_combine(key.members, key.incomplete, key.packed, key.kind,
                              key.ast);
  }

  static StructTypeStorage *construct(mlir::TypeStorageAllocator &allocator,
                                      const KeyTy &key) {
    return new (allocator.allocate<StructTypeStorage>())
        StructTypeStorage(allocator.copyInto(key.members), key.name,
                          key.incomplete, key.packed, key.kind, key.ast);
  }

  /// Mutates the members and attributes an identified struct.
  ///
  /// Once a record is mutated, it is marked as complete, preventing further
  /// mutations. Anonymous structs are always complete and cannot be mutated.
  /// This method does not fail if a mutation of a complete struct does not
  /// change the struct.
  llvm::LogicalResult mutate(mlir::TypeStorageAllocator &allocator,
                             llvm::ArrayRef<mlir::Type> members, bool packed,
                             ASTRecordDeclInterface ast) {
    // Anonymous structs cannot mutate.
    if (!name)
      return llvm::failure();

    // Mutation of complete structs are allowed if they change nothing.
    if (!incomplete)
      return mlir::success((this->members == members) &&
                           (this->packed == packed) && (this->ast == ast));

    // Mutate incomplete struct.
    this->members = allocator.copyInto(members);
    this->packed = packed;
    this->ast = ast;

    incomplete = false;
    return llvm::success();
  }
};

} // namespace detail
} // namespace cir

#endif // CIR_DIALECT_IR_CIRTYPESDETAILS_H
