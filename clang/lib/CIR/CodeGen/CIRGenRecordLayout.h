//===----------------------------------------------------------------------===//
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

namespace clang::CIRGen {

/// This class handles record and union layout info while lowering AST types
/// to CIR types.
///
/// These layout objects are only created on demand as CIR generation requires.
class CIRGenRecordLayout {
  friend class CIRGenTypes;

  CIRGenRecordLayout(const CIRGenRecordLayout &) = delete;
  void operator=(const CIRGenRecordLayout &) = delete;

private:
  /// The CIR type corresponding to this record layout; used when laying it out
  /// as a complete object.
  cir::RecordType completeObjectType;

  /// Map from (non-bit-field) record field to the corresponding cir record type
  /// field no. This info is populated by the record builder.
  llvm::DenseMap<const clang::FieldDecl *, unsigned> fieldIdxMap;

public:
  CIRGenRecordLayout(cir::RecordType completeObjectType)
      : completeObjectType(completeObjectType) {}

  /// Return the "complete object" LLVM type associated with
  /// this record.
  cir::RecordType getCIRType() const { return completeObjectType; }

  /// Return cir::RecordType element number that corresponds to the field FD.
  unsigned getCIRFieldNo(const clang::FieldDecl *fd) const {
    fd = fd->getCanonicalDecl();
    assert(fieldIdxMap.count(fd) && "Invalid field for record!");
    return fieldIdxMap.lookup(fd);
  }
};

} // namespace clang::CIRGen

#endif
