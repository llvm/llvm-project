
#include "CIRDataLayout.h"
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

using namespace llvm;
using namespace clang;
using namespace cir;

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

  /// ----------------------
  /// Short helper routines.

  /// Constructs a MemberInfo instance from an offset and mlir::Type.
  MemberInfo StorageInfo(CharUnits Offset, mlir::Type Data) {
    return MemberInfo(Offset, MemberInfo::InfoKind::Field, Data);
  }

  // Layout routines.
  void setBitFieldInfo(const FieldDecl *FD, CharUnits StartOffset,
                       mlir::Type StorageType);

  void lower(bool nonVirtualBaseType);
  void lowerUnion();

  void computeVolatileBitfields();
  void accumulateBases();
  void accumulateVPtrs();
  void accumulateVBases();
  void accumulateFields();
  void accumulateBitFields(RecordDecl::field_iterator Field,
                           RecordDecl::field_iterator FieldEnd);

  mlir::Type getVFPtrType();

  // Helper function to check if we are targeting AAPCS.
  bool isAAPCS() const {
    return astContext.getTargetInfo().getABI().starts_with("aapcs");
  }

  /// Helper function to check if the target machine is BigEndian.
  bool isBE() const { return astContext.getTargetInfo().isBigEndian(); }

  /// The Microsoft bitfield layout rule allocates discrete storage
  /// units of the field's formal type and only combines adjacent
  /// fields of the same formal type.  We want to emit a layout with
  /// these discrete storage units instead of combining them into a
  /// continuous run.
  bool isDiscreteBitFieldABI() {
    return astContext.getTargetInfo().getCXXABI().isMicrosoft() ||
           recordDecl->isMsStruct(astContext);
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

  CharUnits getSize(mlir::Type Ty) {
    return CharUnits::fromQuantity(dataLayout.layout.getTypeSize(Ty));
  }
  CharUnits getSizeInBits(mlir::Type Ty) {
    return CharUnits::fromQuantity(dataLayout.layout.getTypeSizeInBits(Ty));
  }
  CharUnits getAlignment(mlir::Type Ty) {
    return CharUnits::fromQuantity(dataLayout.layout.getTypeABIAlignment(Ty));
  }
  bool isZeroInitializable(const FieldDecl *FD) {
    return cirGenTypes.isZeroInitializable(FD->getType());
  }
  bool isZeroInitializable(const RecordDecl *RD) {
    return cirGenTypes.isZeroInitializable(RD);
  }

  mlir::Type getCharType() {
    return mlir::cir::IntType::get(&cirGenTypes.getMLIRContext(),
                                   astContext.getCharWidth(),
                                   /*isSigned=*/false);
  }

  /// Wraps mlir::cir::IntType with some implicit arguments.
  mlir::Type getUIntNType(uint64_t NumBits) {
    unsigned AlignedBits = llvm::alignTo(NumBits, astContext.getCharWidth());
    return mlir::cir::IntType::get(&cirGenTypes.getMLIRContext(), AlignedBits,
                                   /*isSigned=*/false);
  }

  mlir::Type getByteArrayType(CharUnits numberOfChars) {
    assert(!numberOfChars.isZero() && "Empty byte arrays aren't allowed.");
    mlir::Type type = getCharType();
    return numberOfChars == CharUnits::One()
               ? type
               : mlir::cir::ArrayType::get(type.getContext(), type,
                                           numberOfChars.getQuantity());
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

    // return getUIntNType(std::min(fielddecl->getBitWidthValue(astContext),
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
  llvm::DenseMap<const FieldDecl *, CIRGenBitFieldInfo> bitFields;
  llvm::DenseMap<const CXXRecordDecl *, unsigned> nonVirtualBases;
  llvm::DenseMap<const CXXRecordDecl *, unsigned> virtualBases;
  CIRDataLayout dataLayout;
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
      dataLayout{cirGenTypes.getModule().getModule()},
      IsZeroInitializable(true), IsZeroInitializableAsBase(true),
      isPacked{isPacked} {}

void CIRRecordLowering::setBitFieldInfo(const FieldDecl *FD,
                                        CharUnits StartOffset,
                                        mlir::Type StorageType) {
  CIRGenBitFieldInfo &Info = bitFields[FD->getCanonicalDecl()];
  Info.IsSigned = FD->getType()->isSignedIntegerOrEnumerationType();
  Info.Offset =
      (unsigned)(getFieldBitOffset(FD) - astContext.toBits(StartOffset));
  Info.Size = FD->getBitWidthValue(astContext);
  Info.StorageSize = getSizeInBits(StorageType).getQuantity();
  Info.StorageOffset = StartOffset;

  if (Info.Size > Info.StorageSize)
    Info.Size = Info.StorageSize;
  // Reverse the bit offsets for big endian machines. Because we represent
  // a bitfield as a single large integer load, we can imagine the bits
  // counting from the most-significant-bit instead of the
  // least-significant-bit.
  if (dataLayout.isBigEndian())
    Info.Offset = Info.StorageSize - (Info.Offset + Info.Size);

  Info.VolatileStorageSize = 0;
  Info.VolatileOffset = 0;
  Info.VolatileStorageOffset = CharUnits::Zero();
}

void CIRRecordLowering::lower(bool nonVirtualBaseType) {
  if (recordDecl->isUnion()) {
    lowerUnion();
    computeVolatileBitfields();
    return;
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

void CIRRecordLowering::lowerUnion() {
  CharUnits LayoutSize = astRecordLayout.getSize();
  mlir::Type StorageType = nullptr;
  bool SeenNamedMember = false;
  // Iterate through the fields setting bitFieldInfo and the Fields array. Also
  // locate the "most appropriate" storage type.  The heuristic for finding the
  // storage type isn't necessary, the first (non-0-length-bitfield) field's
  // type would work fine and be simpler but would be different than what we've
  // been doing and cause lit tests to change.
  for (const auto *Field : recordDecl->fields()) {
    if (Field->isBitField()) {
      if (Field->isZeroLengthBitField(astContext))
        continue;
      llvm_unreachable("NYI");
    }
    fields[Field->getCanonicalDecl()] = 0;
    auto FieldType = getStorageType(Field);
    // Compute zero-initializable status.
    // This union might not be zero initialized: it may contain a pointer to
    // data member which might have some exotic initialization sequence.
    // If this is the case, then we aught not to try and come up with a "better"
    // type, it might not be very easy to come up with a Constant which
    // correctly initializes it.
    if (!SeenNamedMember) {
      SeenNamedMember = Field->getIdentifier();
      if (!SeenNamedMember)
        if (const auto *FieldRD = Field->getType()->getAsRecordDecl())
          SeenNamedMember = FieldRD->findFirstNamedDataMember();
      if (SeenNamedMember && !isZeroInitializable(Field)) {
        IsZeroInitializable = IsZeroInitializableAsBase = false;
        StorageType = FieldType;
      }
    }
    // Because our union isn't zero initializable, we won't be getting a better
    // storage type.
    if (!IsZeroInitializable)
      continue;

    // Conditionally update our storage type if we've got a new "better" one.
    if (!StorageType || getAlignment(FieldType) > getAlignment(StorageType) ||
        (getAlignment(FieldType) == getAlignment(StorageType) &&
         getSize(FieldType) > getSize(StorageType)))
      StorageType = FieldType;
  }
  // If we have no storage type just pad to the appropriate size and return.
  if (!StorageType)
    return appendPaddingBytes(LayoutSize);
  // If our storage size was bigger than our required size (can happen in the
  // case of packed bitfields on Itanium) then just use an I8 array.
  if (LayoutSize < getSize(StorageType))
    StorageType = getByteArrayType(LayoutSize);
  fieldTypes.push_back(StorageType);
  appendPaddingBytes(LayoutSize - getSize(StorageType));
  // Set packed if we need it.
  if (LayoutSize % getAlignment(StorageType))
    isPacked = true;
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
    assert(!UnimplementedFeature::armComputeVolatileBitfields());
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
        setBitFieldInfo(member.fieldDecl, member.offset, fieldTypes.back());
    } else if (member.kind == MemberInfo::InfoKind::Base) {
      nonVirtualBases[member.cxxRecordDecl] = fieldTypes.size() - 1;
    } else if (member.kind == MemberInfo::InfoKind::VBase) {
      llvm_unreachable("NYI");
      // virtualBases[member.cxxRecordDecl] = fieldTypes.size() - 1;
    }
  }
}

void CIRRecordLowering::accumulateBitFields(
    RecordDecl::field_iterator Field, RecordDecl::field_iterator FieldEnd) {
  // Run stores the first element of the current run of bitfields.  FieldEnd is
  // used as a special value to note that we don't have a current run.  A
  // bitfield run is a contiguous collection of bitfields that can be stored in
  // the same storage block.  Zero-sized bitfields and bitfields that would
  // cross an alignment boundary break a run and start a new one.
  RecordDecl::field_iterator Run = FieldEnd;
  // Tail is the offset of the first bit off the end of the current run.  It's
  // used to determine if the ASTRecordLayout is treating these two bitfields as
  // contiguous.  StartBitOffset is offset of the beginning of the Run.
  uint64_t StartBitOffset, Tail = 0;
  if (isDiscreteBitFieldABI()) {
    llvm_unreachable("NYI");
  }

  // Check if OffsetInRecord (the size in bits of the current run) is better
  // as a single field run. When OffsetInRecord has legal integer width, and
  // its bitfield offset is naturally aligned, it is better to make the
  // bitfield a separate storage component so as it can be accessed directly
  // with lower cost.
  auto IsBetterAsSingleFieldRun = [&](uint64_t OffsetInRecord,
                                      uint64_t StartBitOffset) {
    if (!cirGenTypes.getModule().getCodeGenOpts().FineGrainedBitfieldAccesses)
      return false;
    llvm_unreachable("NYI");
    // if (OffsetInRecord < 8 || !llvm::isPowerOf2_64(OffsetInRecord) ||
    //     !DataLayout.fitsInLegalInteger(OffsetInRecord))
    //   return false;
    // Make sure StartBitOffset is naturally aligned if it is treated as an
    // IType integer.
    // if (StartBitOffset %
    //         astContext.toBits(getAlignment(getUIntNType(OffsetInRecord))) !=
    //     0)
    //   return false;
    return true;
  };

  // The start field is better as a single field run.
  bool StartFieldAsSingleRun = false;
  for (;;) {
    // Check to see if we need to start a new run.
    if (Run == FieldEnd) {
      // If we're out of fields, return.
      if (Field == FieldEnd)
        break;
      // Any non-zero-length bitfield can start a new run.
      if (!Field->isZeroLengthBitField(astContext)) {
        Run = Field;
        StartBitOffset = getFieldBitOffset(*Field);
        Tail = StartBitOffset + Field->getBitWidthValue(astContext);
        StartFieldAsSingleRun =
            IsBetterAsSingleFieldRun(Tail - StartBitOffset, StartBitOffset);
      }
      ++Field;
      continue;
    }

    // If the start field of a new run is better as a single run, or if current
    // field (or consecutive fields) is better as a single run, or if current
    // field has zero width bitfield and either UseZeroLengthBitfieldAlignment
    // or UseBitFieldTypeAlignment is set to true, or if the offset of current
    // field is inconsistent with the offset of previous field plus its offset,
    // skip the block below and go ahead to emit the storage. Otherwise, try to
    // add bitfields to the run.
    if (!StartFieldAsSingleRun && Field != FieldEnd &&
        !IsBetterAsSingleFieldRun(Tail - StartBitOffset, StartBitOffset) &&
        (!Field->isZeroLengthBitField(astContext) ||
         (!astContext.getTargetInfo().useZeroLengthBitfieldAlignment() &&
          !astContext.getTargetInfo().useBitFieldTypeAlignment())) &&
        Tail == getFieldBitOffset(*Field)) {
      Tail += Field->getBitWidthValue(astContext);
      ++Field;
      continue;
    }

    // We've hit a break-point in the run and need to emit a storage field.
    auto Type = getUIntNType(Tail - StartBitOffset);
    // Add the storage member to the record and set the bitfield info for all of
    // the bitfields in the run. Bitfields get the offset of their storage but
    // come afterward and remain there after a stable sort.
    members.push_back(StorageInfo(bitsToCharUnits(StartBitOffset), Type));
    for (; Run != Field; ++Run)
      members.push_back(MemberInfo(bitsToCharUnits(StartBitOffset),
                                   MemberInfo::InfoKind::Field, nullptr, *Run));
    Run = FieldEnd;
    StartFieldAsSingleRun = false;
  }
}

void CIRRecordLowering::accumulateFields() {
  for (RecordDecl::field_iterator field = recordDecl->field_begin(),
                                  fieldEnd = recordDecl->field_end();
       field != fieldEnd;) {
    if (field->isBitField()) {
      RecordDecl::field_iterator start = field;
      // Iterate to gather the list of bitfields.
      for (++field; field != fieldEnd && field->isBitField(); ++field)
        ;
      accumulateBitFields(start, field);
    } else if (!field->isZeroSize(astContext)) {
      members.push_back(MemberInfo{bitsToCharUnits(getFieldBitOffset(*field)),
                                   MemberInfo::InfoKind::Field,
                                   getStorageType(*field), *field});
      ++field;
    } else {
      // TODO(cir): do we want to do anything special about zero size
      // members?
      ++field;
    }
  }
}

std::unique_ptr<CIRGenRecordLayout>
CIRGenTypes::computeRecordLayout(const RecordDecl *D,
                                 mlir::cir::StructType *Ty) {
  CIRRecordLowering builder(*this, D, /*packed=*/false);

  builder.lower(/*nonVirtualBaseType=*/false);

  // If we're in C++, compute the base subobject type.
  mlir::cir::StructType *BaseTy = nullptr;
  if (llvm::isa<CXXRecordDecl>(D) && !D->isUnion() &&
      !D->hasAttr<FinalAttr>()) {
    BaseTy = Ty;
    if (builder.astRecordLayout.getNonVirtualSize() !=
        builder.astRecordLayout.getSize()) {
      CIRRecordLowering baseBuilder(*this, D, /*Packed=*/builder.isPacked);
      auto baseIdentifier = getRecordTypeName(D, ".base");
      *BaseTy = Builder.getStructTy(baseBuilder.fieldTypes, baseIdentifier,
                                    /*body=*/true, /*packed=*/false, D);
      // TODO(cir): add something like addRecordTypeName

      // BaseTy and Ty must agree on their packedness for getCIRFieldNo to work
      // on both of them with the same index.
      assert(builder.isPacked == baseBuilder.isPacked &&
             "Non-virtual and complete types must agree on packedness");
    }
  }

  // Fill in the struct *after* computing the base type.  Filling in the body
  // signifies that the type is no longer opaque and record layout is complete,
  // but we may need to recursively layout D while laying D out as a base type.
  *Ty = Builder.getStructTy(builder.fieldTypes, getRecordTypeName(D, ""),
                            /*body=*/true, /*packed=*/false, D);

  auto RL = std::make_unique<CIRGenRecordLayout>(
      Ty ? *Ty : mlir::cir::StructType{},
      BaseTy ? *BaseTy : mlir::cir::StructType{},
      (bool)builder.IsZeroInitializable,
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

CIRGenBitFieldInfo CIRGenBitFieldInfo::MakeInfo(CIRGenTypes &Types,
                                                const FieldDecl *FD,
                                                uint64_t Offset, uint64_t Size,
                                                uint64_t StorageSize,
                                                CharUnits StorageOffset) {
  llvm_unreachable("NYI");
}

CIRDataLayout::CIRDataLayout(mlir::ModuleOp modOp) : layout{modOp} {
  auto dlSpec = modOp->getAttr(mlir::DLTIDialect::kDataLayoutAttrName)
                    .dyn_cast<mlir::DataLayoutSpecAttr>();
  assert(dlSpec && "expected dl_spec in the module");
  auto entries = dlSpec.getEntries();

  for (auto entry : entries) {
    auto entryKey = entry.getKey();
    auto strKey = entryKey.dyn_cast<mlir::StringAttr>();
    if (!strKey)
      continue;
    auto entryName = strKey.strref();
    if (entryName == mlir::DLTIDialect::kDataLayoutEndiannessKey) {
      auto value = entry.getValue().dyn_cast<mlir::StringAttr>();
      assert(value && "expected string attribute");
      auto endian = value.getValue();
      if (endian == mlir::DLTIDialect::kDataLayoutEndiannessBig)
        bigEndian = true;
      else if (endian == mlir::DLTIDialect::kDataLayoutEndiannessLittle)
        bigEndian = false;
      else
        llvm_unreachable("unknown endianess");
    }
  }
}