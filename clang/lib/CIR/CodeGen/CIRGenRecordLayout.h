//===--- CIRGenRecordLayout.h - CIR Record Layout Information ---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_LIB_CIR_CIRGENRECORDLAYOUT_H
#define LLVM_CLANG_LIB_CIR_CIRGENRECORDLAYOUT_H

#include "clang/AST/Decl.h"
#include "clang/CIR/Dialect/IR/CIRTypes.h"

namespace cir {

/// CIRGenRecordLayout - This class handles struct and union layout info while
/// lowering AST types to CIR types.
///
/// These layout objects are only created on demand as CIR generation requires.
class CIRGenRecordLayout {
  friend class CIRGenTypes;

  CIRGenRecordLayout(const CIRGenRecordLayout &) = delete;
  void operator=(const CIRGenRecordLayout &) = delete;

private:
  /// The CIR type corresponding to this record layout; used when laying it out
  /// as a complete object.
  mlir::cir::StructType CompleteObjectType;

  /// The CIR type for the non-virtual part of this record layout; used when
  /// laying it out as a base subobject.
  mlir::cir::StructType BaseSubobjectType;

  /// Map from (non-bit-field) struct field to the corresponding cir struct type
  /// field no. This info is populated by the record builder.
  llvm::DenseMap<const clang::FieldDecl *, unsigned> FieldInfo;

  /// Map from (bit-field) struct field to the corresponding CIR struct type
  /// field no. This info is populated by record builder.
  /// TODO(CIR): value is an int for now, fix when we support bitfields
  llvm::DenseMap<const clang::FieldDecl *, int> BitFields;

  // FIXME: Maybe we could use CXXBaseSpecifier as the key and use a single map
  // for both virtual and non-virtual bases.
  llvm::DenseMap<const clang::CXXRecordDecl *, unsigned> NonVirtualBases;

  /// Map from virtual bases to their field index in the complete object.
  llvm::DenseMap<const clang::CXXRecordDecl *, unsigned>
      CompleteObjectVirtualBases;

  /// False if any direct or indirect subobject of this class, when considered
  /// as a complete object, requires a non-zero bitpattern when
  /// zero-initialized.
  bool IsZeroInitializable : 1;

  /// False if any direct or indirect subobject of this class, when considered
  /// as a base subobject, requires a non-zero bitpattern when zero-initialized.
  bool IsZeroInitializableAsBase : 1;

public:
  CIRGenRecordLayout(mlir::cir::StructType CompleteObjectType,
                     mlir::cir::StructType BaseSubobjectType,
                     bool IsZeroInitializable, bool IsZeroInitializableAsBase)
      : CompleteObjectType(CompleteObjectType),
        BaseSubobjectType(BaseSubobjectType),
        IsZeroInitializable(IsZeroInitializable),
        IsZeroInitializableAsBase(IsZeroInitializableAsBase) {}

  /// Return the "complete object" LLVM type associated with
  /// this record.
  mlir::cir::StructType getCIRType() const { return CompleteObjectType; }

  /// Return the "base subobject" LLVM type associated with
  /// this record.
  mlir::cir::StructType getBaseSubobjectCIRType() const {
    return BaseSubobjectType;
  }

  /// Return cir::StructType element number that corresponds to the field FD.
  unsigned getCIRFieldNo(const clang::FieldDecl *FD) const {
    FD = FD->getCanonicalDecl();
    assert(FieldInfo.count(FD) && "Invalid field for record!");
    return FieldInfo.lookup(FD);
  }

  /// Check whether this struct can be C++ zero-initialized with a
  /// zeroinitializer.
  bool isZeroInitializable() const { return IsZeroInitializable; }
};

} // namespace cir

#endif
