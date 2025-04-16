//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This contains code to compute the layout of a record.
//
//===----------------------------------------------------------------------===//

#include "CIRGenBuilder.h"
#include "CIRGenModule.h"
#include "CIRGenTypes.h"

#include "clang/AST/ASTContext.h"
#include "clang/AST/Decl.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/RecordLayout.h"
#include "clang/CIR/Dialect/IR/CIRAttrs.h"
#include "llvm/Support/Casting.h"

#include <memory>

using namespace llvm;
using namespace clang;
using namespace clang::CIRGen;

namespace {
/// The CIRRecordLowering is responsible for lowering an ASTRecordLayout to an
/// mlir::Type. Some of the lowering is straightforward, some is not.
// TODO: Detail some of the complexities and weirdnesses?
// (See CGRecordLayoutBuilder.cpp)
struct CIRRecordLowering final {

  // MemberInfo is a helper structure that contains information about a record
  // member. In addition to the standard member types, there exists a sentinel
  // member type that ensures correct rounding.
  struct MemberInfo final {
    CharUnits offset;
    enum class InfoKind { Field } kind;
    mlir::Type data;
    union {
      const FieldDecl *fieldDecl;
      // CXXRecordDecl will be used here when base types are supported.
    };
    MemberInfo(CharUnits offset, InfoKind kind, mlir::Type data,
               const FieldDecl *fieldDecl = nullptr)
        : offset(offset), kind(kind), data(data), fieldDecl(fieldDecl) {};
    // MemberInfos are sorted so we define a < operator.
    bool operator<(const MemberInfo &other) const {
      return offset < other.offset;
    }
  };
  // The constructor.
  CIRRecordLowering(CIRGenTypes &cirGenTypes, const RecordDecl *recordDecl,
                    bool isPacked);

  void lower();

  void accumulateFields();

  CharUnits bitsToCharUnits(uint64_t bitOffset) {
    return astContext.toCharUnitsFromBits(bitOffset);
  }

  CharUnits getSize(mlir::Type Ty) {
    assert(!cir::MissingFeatures::recordTypeLayoutInfo());
    return CharUnits::One();
  }
  CharUnits getAlignment(mlir::Type Ty) {
    assert(!cir::MissingFeatures::recordTypeLayoutInfo());
    return CharUnits::One();
  }

  mlir::Type getStorageType(const FieldDecl *fieldDecl) {
    mlir::Type type = cirGenTypes.convertTypeForMem(fieldDecl->getType());
    if (fieldDecl->isBitField()) {
      cirGenTypes.getCGModule().errorNYI(recordDecl->getSourceRange(),
                                         "getStorageType for bitfields");
    }
    return type;
  }

  uint64_t getFieldBitOffset(const FieldDecl *fieldDecl) {
    return astRecordLayout.getFieldOffset(fieldDecl->getFieldIndex());
  }

  /// Fills out the structures that are ultimately consumed.
  void fillOutputFields();

  CIRGenTypes &cirGenTypes;
  CIRGenBuilderTy &builder;
  const ASTContext &astContext;
  const RecordDecl *recordDecl;
  const ASTRecordLayout &astRecordLayout;
  // Helpful intermediate data-structures
  std::vector<MemberInfo> members;
  // Output fields, consumed by CIRGenTypes::computeRecordLayout
  llvm::SmallVector<mlir::Type, 16> fieldTypes;
  llvm::DenseMap<const FieldDecl *, unsigned> fields;

  LLVM_PREFERRED_TYPE(bool)
  unsigned zeroInitializable : 1;
  LLVM_PREFERRED_TYPE(bool)
  unsigned packed : 1;
  LLVM_PREFERRED_TYPE(bool)
  unsigned padded : 1;

private:
  CIRRecordLowering(const CIRRecordLowering &) = delete;
  void operator=(const CIRRecordLowering &) = delete;
}; // CIRRecordLowering
} // namespace

CIRRecordLowering::CIRRecordLowering(CIRGenTypes &cirGenTypes,
                                     const RecordDecl *recordDecl,
                                     bool isPacked)
    : cirGenTypes(cirGenTypes), builder(cirGenTypes.getBuilder()),
      astContext(cirGenTypes.getASTContext()), recordDecl(recordDecl),
      astRecordLayout(
          cirGenTypes.getASTContext().getASTRecordLayout(recordDecl)),
      zeroInitializable(true), packed(isPacked), padded(false) {}

void CIRRecordLowering::lower() {
  if (recordDecl->isUnion()) {
    cirGenTypes.getCGModule().errorNYI(recordDecl->getSourceRange(),
                                       "lower: union");
    return;
  }

  if (isa<CXXRecordDecl>(recordDecl)) {
    cirGenTypes.getCGModule().errorNYI(recordDecl->getSourceRange(),
                                       "lower: class");
    return;
  }

  assert(!cir::MissingFeatures::cxxSupport());

  accumulateFields();

  llvm::stable_sort(members);
  // TODO: implement clipTailPadding once bitfields are implemented
  assert(!cir::MissingFeatures::bitfields());
  // TODO: implemented packed records
  assert(!cir::MissingFeatures::packedRecords());
  // TODO: implement padding
  assert(!cir::MissingFeatures::recordPadding());
  // TODO: support zeroInit
  assert(!cir::MissingFeatures::recordZeroInit());

  fillOutputFields();
}

void CIRRecordLowering::fillOutputFields() {
  for (const MemberInfo &member : members) {
    if (member.data)
      fieldTypes.push_back(member.data);
    if (member.kind == MemberInfo::InfoKind::Field) {
      if (member.fieldDecl)
        fields[member.fieldDecl->getCanonicalDecl()] = fieldTypes.size() - 1;
      // A field without storage must be a bitfield.
      assert(!cir::MissingFeatures::bitfields());
    }
    assert(!cir::MissingFeatures::cxxSupport());
  }
}

void CIRRecordLowering::accumulateFields() {
  for (const FieldDecl *field : recordDecl->fields()) {
    if (field->isBitField()) {
      cirGenTypes.getCGModule().errorNYI(recordDecl->getSourceRange(),
                                         "accumulate bitfields");
      ++field;
    } else if (!field->isZeroSize(astContext)) {
      members.push_back(MemberInfo(bitsToCharUnits(getFieldBitOffset(field)),
                                   MemberInfo::InfoKind::Field,
                                   getStorageType(field), field));
      ++field;
    } else {
      // TODO(cir): do we want to do anything special about zero size members?
      assert(!cir::MissingFeatures::zeroSizeRecordMembers());
      ++field;
    }
  }
}

std::unique_ptr<CIRGenRecordLayout>
CIRGenTypes::computeRecordLayout(const RecordDecl *rd, cir::RecordType *ty) {
  CIRRecordLowering lowering(*this, rd, /*packed=*/false);
  assert(ty->isIncomplete() && "recomputing record layout?");
  lowering.lower();

  // If we're in C++, compute the base subobject type.
  if (llvm::isa<CXXRecordDecl>(rd) && !rd->isUnion() &&
      !rd->hasAttr<FinalAttr>()) {
    cgm.errorNYI(rd->getSourceRange(), "computeRecordLayout: CXXRecordDecl");
  }

  // Fill in the record *after* computing the base type.  Filling in the body
  // signifies that the type is no longer opaque and record layout is complete,
  // but we may need to recursively layout rd while laying D out as a base type.
  assert(!cir::MissingFeatures::astRecordDeclAttr());
  ty->complete(lowering.fieldTypes, lowering.packed, lowering.padded);

  auto rl = std::make_unique<CIRGenRecordLayout>(ty ? *ty : cir::RecordType());

  assert(!cir::MissingFeatures::recordZeroInit());
  assert(!cir::MissingFeatures::cxxSupport());
  assert(!cir::MissingFeatures::bitfields());

  // Add all the field numbers.
  rl->fieldInfo.swap(lowering.fields);

  // Dump the layout, if requested.
  if (getASTContext().getLangOpts().DumpRecordLayouts) {
    cgm.errorNYI(rd->getSourceRange(), "computeRecordLayout: dump layout");
  }

  // TODO: implement verification
  return rl;
}
