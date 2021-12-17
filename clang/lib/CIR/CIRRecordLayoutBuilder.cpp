
#include "CIRGenTypes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/Decl.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/RecordLayout.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/ErrorHandling.h"

#include <memory>

using namespace cir;

namespace {
struct CIRRecordLowering final {

  // MemberInfo is a helper structure that contains information about a record
  // member. In addition to the standard member types, there exists a sentinel
  // member type that ensures correct rounding.
  struct MemberInfo final {
    clang::CharUnits offset;
    enum class InfoKind { VFPtr, VBPtr, Field, Base, VBase, Scissor } kind;
    mlir::Type data;
    const clang::FieldDecl *fieldDecl;
    MemberInfo(clang::CharUnits offset, InfoKind kind, mlir::Type data,
               const clang::FieldDecl *fieldDecl = nullptr)
        : offset{offset}, kind{kind}, data{data}, fieldDecl{fieldDecl} {};
    bool operator<(const MemberInfo &other) const {
      return offset < other.offset;
    }
  };
  CIRRecordLowering(CIRGenTypes &cirGenTypes,
                    const clang::RecordDecl *recordDecl, bool isPacked);

  void lower(bool nonVirtualBaseType);

  void accumulateFields();

  clang::CharUnits bitsToCharUnits(uint64_t bitOffset) {
    return astContext.toCharUnitsFromBits(bitOffset);
  }

  void calculateZeroInit();

  mlir::Type getCharType() {
    return mlir::IntegerType::get(&cirGenTypes.getMLIRContext(),
                                  astContext.getCharWidth());
  }

  mlir::Type getByteArrayType(clang::CharUnits numberOfChars) {
    assert(!numberOfChars.isZero() && "Empty byte arrays aren't allowed.");
    mlir::Type type = getCharType();
    return numberOfChars == clang::CharUnits::One()
               ? type
               : mlir::RankedTensorType::get({0, numberOfChars.getQuantity()},
                                             type);
  }

  mlir::Type getStorageType(const clang::FieldDecl *fieldDecl) {
    auto type = cirGenTypes.convertTypeForMem(fieldDecl->getType());
    assert(!fieldDecl->isBitField() && "bit fields NYI");
    if (!fieldDecl->isBitField())
      return type;

    // if (isDiscreteBitFieldABI())
    //   return type;

    // return getIntNType(std::min(fielddecl->getBitWidthValue(astContext),
    //     static_cast<unsigned int>(astContext.toBits(getSize(type)))));
    llvm_unreachable("getStorageType only supports nonBitFields at this point");
  }

  uint64_t getFieldBitOffset(const clang::FieldDecl *fieldDecl) {
    return astRecordLayout.getFieldOffset(fieldDecl->getFieldIndex());
  }

  /// Fills out the structures that are ultimately consumed.
  void fillOutputFields();

  CIRGenTypes &cirGenTypes;
  const clang::ASTContext &astContext;
  const clang::RecordDecl *recordDecl;
  const clang::CXXRecordDecl *cxxRecordDecl;
  const clang::ASTRecordLayout &astRecordLayout;
  // Helpful intermediate data-structures
  std::vector<MemberInfo> members;
  // Output fields, consumed by CIRGenTypes::computeRecordLayout
  llvm::SmallVector<mlir::Type, 16> fieldTypes;
  llvm::DenseMap<const clang::FieldDecl *, unsigned> fields;
  bool isPacked : 1;

private:
  CIRRecordLowering(const CIRRecordLowering &) = delete;
  void operator=(const CIRRecordLowering &) = delete;
};
} // namespace

CIRRecordLowering::CIRRecordLowering(CIRGenTypes &cirGenTypes,
                                     const clang::RecordDecl *recordDecl,
                                     bool isPacked)
    : cirGenTypes{cirGenTypes}, astContext{cirGenTypes.getContext()},
      recordDecl{recordDecl},
      cxxRecordDecl{llvm::dyn_cast<clang::CXXRecordDecl>(recordDecl)},
      astRecordLayout{cirGenTypes.getContext().getASTRecordLayout(recordDecl)},
      isPacked{isPacked} {}

void CIRRecordLowering::lower(bool nonVirtualBaseType) {
  assert(!recordDecl->isUnion() && "NYI");

  accumulateFields();

  if (cxxRecordDecl) {
    assert(!astRecordLayout.hasOwnVFPtr() && "accumulateVPtrs() NYI");
    assert(cxxRecordDecl->bases().begin() == cxxRecordDecl->bases().end() &&
           "Inheritance NYI");

    assert(!members.empty() && "Empty CXXRecordDecls NYI");
    assert(!nonVirtualBaseType && "non-irtual base type handling NYI");
  }

  llvm::stable_sort(members);
  // TODO: implement clipTailPadding once bitfields are implemented
  // TODO: implemented packed structs
  // TODO: implement padding
  // TODO: support zeroInit
  fillOutputFields();
  // TODO: implement volatile bit fields
}

void CIRRecordLowering::fillOutputFields() {
  for (auto &member : members) {
    assert(member.data && "member.data should be valid");
    fieldTypes.push_back(member.data);
    assert(member.kind == MemberInfo::InfoKind::Field &&
           "Bit fields and inheritance are not NYI");
    assert(member.fieldDecl && "member.fieldDecl should be valid");
    fields[member.fieldDecl->getCanonicalDecl()] = fieldTypes.size() - 1;

    // A field without storage must be a bitfield.
    assert(member.data && "Bitfields NYI");
    assert(member.kind != MemberInfo::InfoKind::Base && "Base classes NYI");
    assert(member.kind != MemberInfo::InfoKind::VBase && "Base classes NYI");
  }
}

void CIRRecordLowering::accumulateFields() {
  for (auto *field : recordDecl->fields()) {
    assert(!field->isBitField() && "bit fields NYI");
    assert(!field->isZeroSize(astContext) && "zero size members NYI");
    members.push_back(MemberInfo{bitsToCharUnits(getFieldBitOffset(field)),
                                 MemberInfo::InfoKind::Field,
                                 getStorageType(field), field});
  }
}

mlir::cir::StructType
CIRGenTypes::computeRecordLayout(const clang::RecordDecl *recordDecl) {
  CIRRecordLowering builder(*this, recordDecl, /*packed=*/false);
  builder.lower(/*nonVirtualBaseType=*/false);

  if (llvm::isa<clang::CXXRecordDecl>(recordDecl)) {
    assert(builder.astRecordLayout.getNonVirtualSize() ==
               builder.astRecordLayout.getSize() &&
           "Virtual base objects NYI");
  }

  assert(!builder.isPacked && "Packed structs NYI");

  auto name = getRecordTypeName(recordDecl, "");
  auto identifier = mlir::StringAttr::get(&getMLIRContext(), name);
  auto structType = mlir::cir::StructType::get(&getMLIRContext(),
                                               builder.fieldTypes, identifier);

  assert(!getContext().getLangOpts().DumpRecordLayouts &&
         "RecordLayouts dumping NYI");

  // TODO: implement verification

  return structType;
}
