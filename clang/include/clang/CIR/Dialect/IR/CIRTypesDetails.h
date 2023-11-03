//===- CIRTypesDetails.h - Details of CIR dialect types -----------*- C++ -*-===//
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
#include "clang/CIR/Dialect/IR/CIRTypes.h"

namespace mlir {
namespace cir {
namespace detail {

//===----------------------------------------------------------------------===//
// CIR StructTypeStorage
//===----------------------------------------------------------------------===//

/// Type storage for CIR record types.
struct StructTypeStorage : public TypeStorage {
  struct KeyTy {
    ArrayRef<Type> members;
    StringAttr name;
    bool incomplete;
    bool packed;
    StructType::RecordKind kind;
    ASTRecordDeclInterface ast;

    KeyTy(ArrayRef<Type> members, StringAttr name, bool incomplete, bool packed,
          StructType::RecordKind kind, ASTRecordDeclInterface ast)
        : members(members), name(name), incomplete(incomplete), packed(packed),
          kind(kind), ast(ast) {}
  };

  ArrayRef<Type> members;
  StringAttr name;
  bool incomplete;
  bool packed;
  StructType::RecordKind kind;
  ASTRecordDeclInterface ast;

  StructTypeStorage(ArrayRef<Type> members, StringAttr name, bool incomplete,
                    bool packed, StructType::RecordKind kind,
                    ASTRecordDeclInterface ast)
      : members(members), name(name), incomplete(incomplete), packed(packed),
        kind(kind), ast(ast) {}

  KeyTy getAsKey() const {
    return KeyTy(members, name, incomplete, packed, kind, ast);
  }

  bool operator==(const KeyTy &key) const {
    return (members == key.members) && (name == key.name) &&
           (incomplete == key.incomplete) && (packed == key.packed) &&
           (kind == key.kind) && (ast == key.ast);
  }

  static llvm::hash_code hashKey(const KeyTy &key) {
    return hash_combine(key.members, key.name, key.incomplete, key.packed,
                        key.kind, key.ast);
  }

  static StructTypeStorage *construct(TypeStorageAllocator &allocator,
                                      const KeyTy &key) {
    return new (allocator.allocate<StructTypeStorage>())
        StructTypeStorage(allocator.copyInto(key.members), key.name,
                          key.incomplete, key.packed, key.kind, key.ast);
  }
};

} // namespace detail
} // namespace cir
} // namespace mlir

#endif // CIR_DIALECT_IR_CIRTYPESDETAILS_H
