
#include "CIRGenTypes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/Decl.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/RecordLayout.h"
#include "clang/CIR/Dialect/IR/CIRAttrs.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/ErrorHandling.h"

#include <memory>

using namespace cir;
using namespace clang;

namespace {
/// The CIRRecordLowering is responsible for lowering an ASTRecordLayout to a
/// mlir::Type. Some of the lowering is straightforward, some is not. Here we
/// detail some of the complexities and weirdnesses here.
struct CIRRecordLowering final {

  // MemberInfo is a helper structure that contains information about a record
  // member. In addition to the standard member types, there exists a sentinel
  // member type that ensures correct rounding.
  struct MemberInfo final {
    CharUnits offset;
    enum class InfoKind { VFPtr, VBPtr, Field, Base, VBase, Scissor } kind;
    mlir::Type data;
    const FieldDecl *fieldDecl;
    MemberInfo(CharUnits offset, InfoKind kind, mlir::Type data,
               const FieldDecl *fieldDecl = nullptr)
        : offset{offset}, kind{kind}, data{data}, fieldDecl{fieldDecl} {};
    // MemberInfos are sorted so we define a < operator.
    bool operator<(const MemberInfo &other) const {
      return offset < other.offset;
    }
  };
  // The constructor.
  CIRRecordLowering(CIRGenTypes &cirGenTypes, const RecordDecl *recordDecl,
                    bool isPacked);
  // Short helper routines.

  void lower(bool nonVirtualBaseType);

  void accumulateFields();
  void accumulateVBases();

  CharUnits bitsToCharUnits(uint64_t bitOffset) {
    return astContext.toCharUnitsFromBits(bitOffset);
  }

  void calculateZeroInit();

  mlir::Type getCharType() {
    return mlir::IntegerType::get(&cirGenTypes.getMLIRContext(),
                                  astContext.getCharWidth());
  }

  mlir::Type getByteArrayType(CharUnits numberOfChars) {
    assert(!numberOfChars.isZero() && "Empty byte arrays aren't allowed.");
    mlir::Type type = getCharType();
    return numberOfChars == CharUnits::One()
               ? type
               : mlir::RankedTensorType::get({0, numberOfChars.getQuantity()},
                                             type);
  }

  mlir::Type getStorageType(const FieldDecl *fieldDecl) {
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

  uint64_t getFieldBitOffset(const FieldDecl *fieldDecl) {
    return astRecordLayout.getFieldOffset(fieldDecl->getFieldIndex());
  }

  /// Fills out the structures that are ultimately consumed.
  void fillOutputFields();

  void appendPaddingBytes(CharUnits Size) {
    if (!Size.isZero())
      fieldTypes.push_back(getByteArrayType(Size));
  }

  CIRGenTypes &cirGenTypes;
  const ASTContext &astContext;
  const RecordDecl *recordDecl;
  const CXXRecordDecl *cxxRecordDecl;
  const ASTRecordLayout &astRecordLayout;
  // Helpful intermediate data-structures
  std::vector<MemberInfo> members;
  // Output fields, consumed by CIRGenTypes::computeRecordLayout
  llvm::SmallVector<mlir::Type, 16> fieldTypes;
  llvm::DenseMap<const FieldDecl *, unsigned> fields;
  llvm::DenseMap<const FieldDecl *, int> bitFields;
  llvm::DenseMap<const CXXRecordDecl *, unsigned> nonVirtualBases;
  llvm::DenseMap<const CXXRecordDecl *, unsigned> virtualBases;
  bool IsZeroInitializable : 1;
  bool IsZeroInitializableAsBase : 1;
  bool isPacked : 1;

private:
  CIRRecordLowering(const CIRRecordLowering &) = delete;
  void operator=(const CIRRecordLowering &) = delete;
};
} // namespace

CIRRecordLowering::CIRRecordLowering(CIRGenTypes &cirGenTypes,
                                     const RecordDecl *recordDecl,
                                     bool isPacked)
    : cirGenTypes{cirGenTypes}, astContext{cirGenTypes.getContext()},
      recordDecl{recordDecl},
      cxxRecordDecl{llvm::dyn_cast<CXXRecordDecl>(recordDecl)},
      astRecordLayout{cirGenTypes.getContext().getASTRecordLayout(recordDecl)},
      IsZeroInitializable(true), IsZeroInitializableAsBase(true),
      isPacked{isPacked} {}

void CIRRecordLowering::lower(bool nonVirtualBaseType) {
  if (recordDecl->isUnion()) {
    llvm_unreachable("NYI");
  }

  CharUnits Size = nonVirtualBaseType ? astRecordLayout.getNonVirtualSize()
                                      : astRecordLayout.getSize();

  accumulateFields();

  // RD implies C++
  if (cxxRecordDecl) {
    assert(!astRecordLayout.hasOwnVFPtr() && "accumulateVPtrs() NYI");
    assert(cxxRecordDecl->bases().begin() == cxxRecordDecl->bases().end() &&
           "Inheritance NYI");

    if (members.empty()) {
      appendPaddingBytes(Size);
      // TODO: computeVolatileBitFields();
      return;
    }

    if (!nonVirtualBaseType)
      accumulateVBases();
  }

  llvm::stable_sort(members);
  // TODO: implement clipTailPadding once bitfields are implemented
  // TODO: implemented packed structs
  // TODO: implement padding
  // TODO: support zeroInit
  fillOutputFields();
  // TODO: implement volatile bit fields
}

void CIRRecordLowering::accumulateVBases() {
  if (astRecordLayout.hasOwnVFPtr())
    llvm_unreachable("NYI");
  if (astRecordLayout.hasOwnVBPtr())
    llvm_unreachable("NYI");
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

std::unique_ptr<CIRGenRecordLayout>
CIRGenTypes::computeRecordLayout(const RecordDecl *D,
                                 mlir::cir::StructType &Ty) {
  CIRRecordLowering builder(*this, D, /*packed=*/false);

  builder.lower(/*nonVirtualBaseType=*/false);

  auto name = getRecordTypeName(D, "");
  auto identifier = mlir::StringAttr::get(&getMLIRContext(), name);

  // If we're in C++, compute the base subobject type.
  mlir::cir::StructType BaseTy = nullptr;
  if (llvm::isa<CXXRecordDecl>(D) && !D->isUnion() &&
      !D->hasAttr<FinalAttr>()) {
    BaseTy = Ty;
    if (builder.astRecordLayout.getNonVirtualSize() !=
        builder.astRecordLayout.getSize()) {
      CIRRecordLowering baseBuilder(*this, D, /*Packed=*/builder.isPacked);
      auto baseIdentifier =
          mlir::StringAttr::get(&getMLIRContext(), name + ".base");
      BaseTy = mlir::cir::StructType::get(
          &getMLIRContext(), baseBuilder.fieldTypes, baseIdentifier,
          /*body=*/true,
          mlir::cir::ASTRecordDeclAttr::get(&getMLIRContext(), D));
      // BaseTy and Ty must agree on their packedness for getCIRFieldNo to work
      // on both of them with the same index.
      assert(builder.isPacked == baseBuilder.isPacked &&
             "Non-virtual and complete types must agree on packedness");
    }
  }

  // TODO(cir): add base class info
  Ty = mlir::cir::StructType::get(
      &getMLIRContext(), builder.fieldTypes, identifier,
      /*body=*/true, mlir::cir::ASTRecordDeclAttr::get(&getMLIRContext(), D));

  auto RL = std::make_unique<CIRGenRecordLayout>(
      Ty, BaseTy, (bool)builder.IsZeroInitializable,
      (bool)builder.IsZeroInitializableAsBase);

  RL->NonVirtualBases.swap(builder.nonVirtualBases);
  RL->CompleteObjectVirtualBases.swap(builder.virtualBases);

  // Add all the field numbers.
  RL->FieldInfo.swap(builder.fields);

  // Add bitfield info.
  RL->BitFields.swap(builder.bitFields);

  // Dump the layout, if requested.
  if (getContext().getLangOpts().DumpRecordLayouts) {
    llvm_unreachable("NYI");
  }

  // TODO: implement verification

  return RL;
}
