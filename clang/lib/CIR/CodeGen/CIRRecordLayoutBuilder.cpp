
#include "CIRGenBuilder.h"
#include "CIRGenModule.h"
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
/// mlir::Type. Some of the lowering is straightforward, some is not. TODO: Here
/// we detail some of the complexities and weirdnesses?
struct CIRRecordLowering final {

  // MemberInfo is a helper structure that contains information about a record
  // member. In addition to the standard member types, there exists a sentinel
  // member type that ensures correct rounding.
  struct MemberInfo final {
    CharUnits offset;
    enum class InfoKind { VFPtr, VBPtr, Field, Base, VBase, Scissor } kind;
    mlir::Type data;
    union {
      const FieldDecl *fieldDecl;
      const CXXRecordDecl *cxxRecordDecl;
    };
    MemberInfo(CharUnits offset, InfoKind kind, mlir::Type data,
               const FieldDecl *fieldDecl = nullptr)
        : offset{offset}, kind{kind}, data{data}, fieldDecl{fieldDecl} {};
    MemberInfo(CharUnits offset, InfoKind kind, mlir::Type data,
               const CXXRecordDecl *RD)
        : offset{offset}, kind{kind}, data{data}, cxxRecordDecl{RD} {}
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

  void computeVolatileBitfields();
  void accumulateBases();
  void accumulateVPtrs();
  void accumulateVBases();
  void accumulateFields();

  mlir::Type getVFPtrType();

  // Helper function to check if we are targeting AAPCS.
  bool isAAPCS() const {
    return astContext.getTargetInfo().getABI().starts_with("aapcs");
  }

  // The Itanium base layout rule allows virtual bases to overlap
  // other bases, which complicates layout in specific ways.
  //
  // Note specifically that the ms_struct attribute doesn't change this.
  bool isOverlappingVBaseABI() {
    return !astContext.getTargetInfo().getCXXABI().isMicrosoft();
  }
  // Recursively searches all of the bases to find out if a vbase is
  // not the primary vbase of some base class.
  bool hasOwnStorage(const CXXRecordDecl *Decl, const CXXRecordDecl *Query);

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

  // Gets the llvm Basesubobject type from a CXXRecordDecl.
  mlir::Type getStorageType(const CXXRecordDecl *RD) {
    return cirGenTypes.getCIRGenRecordLayout(RD).getBaseSubobjectCIRType();
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
  CIRGenBuilderTy &builder;
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
    : cirGenTypes{cirGenTypes}, builder{cirGenTypes.getBuilder()},
      astContext{cirGenTypes.getContext()}, recordDecl{recordDecl},
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
  if (recordDecl->isUnion()) {
    llvm_unreachable("NYI");
    // lowerUnion();
    // computeVolatileBitfields();
    return;
  }
  accumulateFields();

  // RD implies C++
  if (cxxRecordDecl) {
    accumulateVPtrs();
    accumulateBases();
    if (members.empty()) {
      appendPaddingBytes(Size);
      computeVolatileBitfields();
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
  computeVolatileBitfields();
}

bool CIRRecordLowering::hasOwnStorage(const CXXRecordDecl *Decl,
                                      const CXXRecordDecl *Query) {
  const ASTRecordLayout &DeclLayout = astContext.getASTRecordLayout(Decl);
  if (DeclLayout.isPrimaryBaseVirtual() && DeclLayout.getPrimaryBase() == Query)
    return false;
  for (const auto &Base : Decl->bases())
    if (!hasOwnStorage(Base.getType()->getAsCXXRecordDecl(), Query))
      return false;
  return true;
}

/// The AAPCS that defines that, when possible, bit-fields should
/// be accessed using containers of the declared type width:
/// When a volatile bit-field is read, and its container does not overlap with
/// any non-bit-field member or any zero length bit-field member, its container
/// must be read exactly once using the access width appropriate to the type of
/// the container. When a volatile bit-field is written, and its container does
/// not overlap with any non-bit-field member or any zero-length bit-field
/// member, its container must be read exactly once and written exactly once
/// using the access width appropriate to the type of the container. The two
/// accesses are not atomic.
///
/// Enforcing the width restriction can be disabled using
/// -fno-aapcs-bitfield-width.
void CIRRecordLowering::computeVolatileBitfields() {
  if (!isAAPCS() ||
      !cirGenTypes.getModule().getCodeGenOpts().AAPCSBitfieldWidth)
    return;

  for ([[maybe_unused]] auto &I : bitFields) {
    llvm_unreachable("NYI");
  }
}

void CIRRecordLowering::accumulateBases() {
  // If we've got a primary virtual base, we need to add it with the bases.
  if (astRecordLayout.isPrimaryBaseVirtual()) {
    llvm_unreachable("NYI");
  }

  // Accumulate the non-virtual bases.
  for ([[maybe_unused]] const auto &Base : cxxRecordDecl->bases()) {
    if (Base.isVirtual())
      continue;
    // Bases can be zero-sized even if not technically empty if they
    // contain only a trailing array member.
    const CXXRecordDecl *BaseDecl = Base.getType()->getAsCXXRecordDecl();
    if (!BaseDecl->isEmpty() &&
        !astContext.getASTRecordLayout(BaseDecl).getNonVirtualSize().isZero()) {
      members.push_back(MemberInfo(astRecordLayout.getBaseClassOffset(BaseDecl),
                                   MemberInfo::InfoKind::Base,
                                   getStorageType(BaseDecl), BaseDecl));
    }
  }
}

void CIRRecordLowering::accumulateVBases() {
  CharUnits ScissorOffset = astRecordLayout.getNonVirtualSize();
  // In the itanium ABI, it's possible to place a vbase at a dsize that is
  // smaller than the nvsize.  Here we check to see if such a base is placed
  // before the nvsize and set the scissor offset to that, instead of the
  // nvsize.
  if (isOverlappingVBaseABI())
    for (const auto &Base : cxxRecordDecl->vbases()) {
      const CXXRecordDecl *BaseDecl = Base.getType()->getAsCXXRecordDecl();
      if (BaseDecl->isEmpty())
        continue;
      llvm_unreachable("NYI");
    }
  members.push_back(MemberInfo(ScissorOffset, MemberInfo::InfoKind::Scissor,
                               mlir::Type{}, cxxRecordDecl));
  for (const auto &Base : cxxRecordDecl->vbases()) {
    const CXXRecordDecl *BaseDecl = Base.getType()->getAsCXXRecordDecl();
    if (BaseDecl->isEmpty())
      continue;
    llvm_unreachable("NYI");
  }
}

void CIRRecordLowering::accumulateVPtrs() {
  if (astRecordLayout.hasOwnVFPtr())
    members.push_back(MemberInfo(CharUnits::Zero(), MemberInfo::InfoKind::VFPtr,
                                 getVFPtrType()));
  if (astRecordLayout.hasOwnVBPtr())
    llvm_unreachable("NYI");
}

mlir::Type CIRRecordLowering::getVFPtrType() {
  // FIXME: replay LLVM codegen for now, perhaps add a vtable ptr special
  // type so it's a bit more clear and C++ idiomatic.
  return builder.getVirtualFnPtrType();
}

void CIRRecordLowering::fillOutputFields() {
  for (auto &member : members) {
    if (member.data)
      fieldTypes.push_back(member.data);
    if (member.kind == MemberInfo::InfoKind::Field) {
      if (member.fieldDecl)
        fields[member.fieldDecl->getCanonicalDecl()] = fieldTypes.size() - 1;
      // A field without storage must be a bitfield.
      if (!member.data)
        llvm_unreachable("NYI");
    } else if (member.kind == MemberInfo::InfoKind::Base) {
      nonVirtualBases[member.cxxRecordDecl] = fieldTypes.size() - 1;
    } else if (member.kind == MemberInfo::InfoKind::VBase) {
      llvm_unreachable("NYI");
      // virtualBases[member.cxxRecordDecl] = fieldTypes.size() - 1;
    }
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
