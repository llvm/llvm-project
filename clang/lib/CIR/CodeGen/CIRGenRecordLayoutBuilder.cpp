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
#include "clang/CIR/Dialect/IR/CIRDataLayout.h"
#include "clang/CIR/MissingFeatures.h"
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
    enum class InfoKind { VFPtr, Field, Base, VBase } kind;
    mlir::Type data;
    union {
      const FieldDecl *fieldDecl;
      const CXXRecordDecl *cxxRecordDecl;
    };
    MemberInfo(CharUnits offset, InfoKind kind, mlir::Type data,
               const FieldDecl *fieldDecl = nullptr)
        : offset{offset}, kind{kind}, data{data}, fieldDecl{fieldDecl} {}
    MemberInfo(CharUnits offset, InfoKind kind, mlir::Type data,
               const CXXRecordDecl *rd)
        : offset{offset}, kind{kind}, data{data}, cxxRecordDecl{rd} {}
    // MemberInfos are sorted so we define a < operator.
    bool operator<(const MemberInfo &other) const {
      return offset < other.offset;
    }
  };
  // The constructor.
  CIRRecordLowering(CIRGenTypes &cirGenTypes, const RecordDecl *recordDecl,
                    bool packed);

  /// Constructs a MemberInfo instance from an offset and mlir::Type.
  MemberInfo makeStorageInfo(CharUnits offset, mlir::Type data) {
    return MemberInfo(offset, MemberInfo::InfoKind::Field, data);
  }

  // Layout routines.
  void setBitFieldInfo(const FieldDecl *fd, CharUnits startOffset,
                       mlir::Type storageType);

  void lower(bool NonVirtualBaseType);
  void lowerUnion();

  /// Determines if we need a packed llvm struct.
  void determinePacked(bool nvBaseType);
  /// Inserts padding everywhere it's needed.
  void insertPadding();

  void computeVolatileBitfields();
  void accumulateBases();
  void accumulateVPtrs();
  void accumulateVBases();
  void accumulateFields();
  RecordDecl::field_iterator
  accumulateBitFields(RecordDecl::field_iterator field,
                      RecordDecl::field_iterator fieldEnd);

  mlir::Type getVFPtrType();

  bool isAAPCS() const {
    return astContext.getTargetInfo().getABI().starts_with("aapcs");
  }

  /// Helper function to check if the target machine is BigEndian.
  bool isBigEndian() const { return astContext.getTargetInfo().isBigEndian(); }

  // The Itanium base layout rule allows virtual bases to overlap
  // other bases, which complicates layout in specific ways.
  //
  // Note specifically that the ms_struct attribute doesn't change this.
  bool isOverlappingVBaseABI() {
    return !astContext.getTargetInfo().getCXXABI().isMicrosoft();
  }
  // Recursively searches all of the bases to find out if a vbase is
  // not the primary vbase of some base class.
  bool hasOwnStorage(const CXXRecordDecl *decl, const CXXRecordDecl *query);

  CharUnits bitsToCharUnits(uint64_t bitOffset) {
    return astContext.toCharUnitsFromBits(bitOffset);
  }

  void calculateZeroInit();

  CharUnits getSize(mlir::Type Ty) {
    return CharUnits::fromQuantity(dataLayout.layout.getTypeSize(Ty));
  }
  CharUnits getSizeInBits(mlir::Type ty) {
    return CharUnits::fromQuantity(dataLayout.layout.getTypeSizeInBits(ty));
  }
  CharUnits getAlignment(mlir::Type Ty) {
    return CharUnits::fromQuantity(dataLayout.layout.getTypeABIAlignment(Ty));
  }

  bool isZeroInitializable(const FieldDecl *fd) {
    return cirGenTypes.isZeroInitializable(fd->getType());
  }
  bool isZeroInitializable(const RecordDecl *rd) {
    return cirGenTypes.isZeroInitializable(rd);
  }

  /// Wraps cir::IntType with some implicit arguments.
  mlir::Type getUIntNType(uint64_t numBits) {
    unsigned alignedBits = llvm::PowerOf2Ceil(numBits);
    alignedBits = std::max(8u, alignedBits);
    return cir::IntType::get(&cirGenTypes.getMLIRContext(), alignedBits,
                             /*isSigned=*/false);
  }

  mlir::Type getCharType() {
    return cir::IntType::get(&cirGenTypes.getMLIRContext(),
                             astContext.getCharWidth(),
                             /*isSigned=*/false);
  }

  mlir::Type getByteArrayType(CharUnits numberOfChars) {
    assert(!numberOfChars.isZero() && "Empty byte arrays aren't allowed.");
    mlir::Type type = getCharType();
    return numberOfChars == CharUnits::One()
               ? type
               : cir::ArrayType::get(type, numberOfChars.getQuantity());
  }

  // Gets the CIR BaseSubobject type from a CXXRecordDecl.
  mlir::Type getStorageType(const CXXRecordDecl *RD) {
    return cirGenTypes.getCIRGenRecordLayout(RD).getBaseSubobjectCIRType();
  }
  // This is different from LLVM traditional codegen because CIRGen uses arrays
  // of bytes instead of arbitrary-sized integers. This is important for packed
  // structures support.
  mlir::Type getBitfieldStorageType(unsigned numBits) {
    unsigned alignedBits = llvm::alignTo(numBits, astContext.getCharWidth());
    if (cir::isValidFundamentalIntWidth(alignedBits))
      return builder.getUIntNTy(alignedBits);

    mlir::Type type = getCharType();
    return cir::ArrayType::get(type, alignedBits / astContext.getCharWidth());
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

  void appendPaddingBytes(CharUnits size) {
    if (!size.isZero()) {
      fieldTypes.push_back(getByteArrayType(size));
      padded = true;
    }
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
  llvm::DenseMap<const FieldDecl *, CIRGenBitFieldInfo> bitFields;
  llvm::DenseMap<const FieldDecl *, unsigned> fieldIdxMap;
  llvm::DenseMap<const CXXRecordDecl *, unsigned> nonVirtualBases;
  llvm::DenseMap<const CXXRecordDecl *, unsigned> virtualBases;
  cir::CIRDataLayout dataLayout;

  LLVM_PREFERRED_TYPE(bool)
  unsigned zeroInitializable : 1;
  LLVM_PREFERRED_TYPE(bool)
  unsigned zeroInitializableAsBase : 1;
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
                                     const RecordDecl *recordDecl, bool packed)
    : cirGenTypes{cirGenTypes}, builder{cirGenTypes.getBuilder()},
      astContext{cirGenTypes.getASTContext()}, recordDecl{recordDecl},
      cxxRecordDecl{llvm::dyn_cast<CXXRecordDecl>(recordDecl)},
      astRecordLayout{
          cirGenTypes.getASTContext().getASTRecordLayout(recordDecl)},
      dataLayout{cirGenTypes.getCGModule().getModule()},
      zeroInitializable{true}, zeroInitializableAsBase{true}, packed{packed},
      padded{false} {}

void CIRRecordLowering::setBitFieldInfo(const FieldDecl *fd,
                                        CharUnits startOffset,
                                        mlir::Type storageType) {
  CIRGenBitFieldInfo &info = bitFields[fd->getCanonicalDecl()];
  info.isSigned = fd->getType()->isSignedIntegerOrEnumerationType();
  info.offset =
      (unsigned)(getFieldBitOffset(fd) - astContext.toBits(startOffset));
  info.size = fd->getBitWidthValue();
  info.storageSize = getSizeInBits(storageType).getQuantity();
  info.storageOffset = startOffset;
  info.storageType = storageType;
  info.name = fd->getName();

  if (info.size > info.storageSize)
    info.size = info.storageSize;
  // Reverse the bit offsets for big endian machines. Since bitfields are laid
  // out as packed bits within an integer-sized unit, we can imagine the bits
  // counting from the most-significant-bit instead of the
  // least-significant-bit.
  if (dataLayout.isBigEndian())
    info.offset = info.storageSize - (info.offset + info.size);

  info.volatileStorageSize = 0;
  info.volatileOffset = 0;
  info.volatileStorageOffset = CharUnits::Zero();
}

void CIRRecordLowering::lower(bool nonVirtualBaseType) {
  if (recordDecl->isUnion()) {
    lowerUnion();
    computeVolatileBitfields();
    return;
  }

  CharUnits size = nonVirtualBaseType ? astRecordLayout.getNonVirtualSize()
                                      : astRecordLayout.getSize();

  accumulateFields();

  if (cxxRecordDecl) {
    accumulateVPtrs();
    accumulateBases();
    if (members.empty()) {
      appendPaddingBytes(size);
      computeVolatileBitfields();
      return;
    }
    if (!nonVirtualBaseType)
      accumulateVBases();
  }

  llvm::stable_sort(members);
  // TODO: implement clipTailPadding once bitfields are implemented
  assert(!cir::MissingFeatures::bitfields());
  assert(!cir::MissingFeatures::recordZeroInit());

  members.push_back(makeStorageInfo(size, getUIntNType(8)));
  determinePacked(nonVirtualBaseType);
  insertPadding();
  members.pop_back();

  calculateZeroInit();
  fillOutputFields();
  computeVolatileBitfields();
}

void CIRRecordLowering::fillOutputFields() {
  for (const MemberInfo &member : members) {
    if (member.data)
      fieldTypes.push_back(member.data);
    if (member.kind == MemberInfo::InfoKind::Field) {
      if (member.fieldDecl)
        fieldIdxMap[member.fieldDecl->getCanonicalDecl()] =
            fieldTypes.size() - 1;
      // A field without storage must be a bitfield.
      assert(!cir::MissingFeatures::bitfields());
      if (!member.data)
        setBitFieldInfo(member.fieldDecl, member.offset, fieldTypes.back());
    } else if (member.kind == MemberInfo::InfoKind::Base) {
      nonVirtualBases[member.cxxRecordDecl] = fieldTypes.size() - 1;
    } else if (member.kind == MemberInfo::InfoKind::VBase) {
      virtualBases[member.cxxRecordDecl] = fieldTypes.size() - 1;
    }
  }
}

RecordDecl::field_iterator
CIRRecordLowering::accumulateBitFields(RecordDecl::field_iterator field,
                                       RecordDecl::field_iterator fieldEnd) {
  assert(!cir::MissingFeatures::isDiscreteBitFieldABI());

  CharUnits regSize =
      bitsToCharUnits(astContext.getTargetInfo().getRegisterWidth());
  unsigned charBits = astContext.getCharWidth();

  // Data about the start of the span we're accumulating to create an access
  // unit from. 'Begin' is the first bitfield of the span. If 'begin' is
  // 'fieldEnd', we've not got a current span. The span starts at the
  // 'beginOffset' character boundary. 'bitSizeSinceBegin' is the size (in bits)
  // of the span -- this might include padding when we've advanced to a
  // subsequent bitfield run.
  RecordDecl::field_iterator begin = fieldEnd;
  CharUnits beginOffset;
  uint64_t bitSizeSinceBegin;

  // The (non-inclusive) end of the largest acceptable access unit we've found
  // since 'begin'. If this is 'begin', we're gathering the initial set of
  // bitfields of a new span. 'bestEndOffset' is the end of that acceptable
  // access unit -- it might extend beyond the last character of the bitfield
  // run, using available padding characters.
  RecordDecl::field_iterator bestEnd = begin;
  CharUnits bestEndOffset;
  bool bestClipped; // Whether the representation must be in a byte array.

  for (;;) {
    // atAlignedBoundary is true if 'field' is the (potential) start of a new
    // span (or the end of the bitfields). When true, limitOffset is the
    // character offset of that span and barrier indicates whether the new
    // span cannot be merged into the current one.
    bool atAlignedBoundary = false;
    bool barrier = false; // a barrier can be a zero Bit Width or non bit member
    if (field != fieldEnd && field->isBitField()) {
      uint64_t bitOffset = getFieldBitOffset(*field);
      if (begin == fieldEnd) {
        // Beginning a new span.
        begin = field;
        bestEnd = begin;

        assert((bitOffset % charBits) == 0 && "Not at start of char");
        beginOffset = bitsToCharUnits(bitOffset);
        bitSizeSinceBegin = 0;
      } else if ((bitOffset % charBits) != 0) {
        // Bitfield occupies the same character as previous bitfield, it must be
        // part of the same span. This can include zero-length bitfields, should
        // the target not align them to character boundaries. Such non-alignment
        // is at variance with the standards, which require zero-length
        // bitfields be a barrier between access units. But of course we can't
        // achieve that in the middle of a character.
        assert(bitOffset ==
                   astContext.toBits(beginOffset) + bitSizeSinceBegin &&
               "Concatenating non-contiguous bitfields");
      } else {
        // Bitfield potentially begins a new span. This includes zero-length
        // bitfields on non-aligning targets that lie at character boundaries
        // (those are barriers to merging).
        if (field->isZeroLengthBitField())
          barrier = true;
        atAlignedBoundary = true;
      }
    } else {
      // We've reached the end of the bitfield run. Either we're done, or this
      // is a barrier for the current span.
      if (begin == fieldEnd)
        break;

      barrier = true;
      atAlignedBoundary = true;
    }

    // 'installBest' indicates whether we should create an access unit for the
    // current best span: fields ['begin', 'bestEnd') occupying characters
    // ['beginOffset', 'bestEndOffset').
    bool installBest = false;
    if (atAlignedBoundary) {
      // 'field' is the start of a new span or the end of the bitfields. The
      // just-seen span now extends to 'bitSizeSinceBegin'.

      // Determine if we can accumulate that just-seen span into the current
      // accumulation.
      CharUnits accessSize = bitsToCharUnits(bitSizeSinceBegin + charBits - 1);
      if (bestEnd == begin) {
        // This is the initial run at the start of a new span. By definition,
        // this is the best seen so far.
        bestEnd = field;
        bestEndOffset = beginOffset + accessSize;
        // Assume clipped until proven not below.
        bestClipped = true;
        if (!bitSizeSinceBegin)
          // A zero-sized initial span -- this will install nothing and reset
          // for another.
          installBest = true;
      } else if (accessSize > regSize) {
        // Accumulating the just-seen span would create a multi-register access
        // unit, which would increase register pressure.
        installBest = true;
      }

      if (!installBest) {
        // Determine if accumulating the just-seen span will create an expensive
        // access unit or not.
        mlir::Type type = getUIntNType(astContext.toBits(accessSize));
        if (!astContext.getTargetInfo().hasCheapUnalignedBitFieldAccess())
          cirGenTypes.getCGModule().errorNYI(
              field->getSourceRange(), "NYI CheapUnalignedBitFieldAccess");

        if (!installBest) {
          // Find the next used storage offset to determine what the limit of
          // the current span is. That's either the offset of the next field
          // with storage (which might be field itself) or the end of the
          // non-reusable tail padding.
          CharUnits limitOffset;
          for (auto probe = field; probe != fieldEnd; ++probe)
            if (!isEmptyFieldForLayout(astContext, *probe)) {
              // A member with storage sets the limit.
              assert((getFieldBitOffset(*probe) % charBits) == 0 &&
                     "Next storage is not byte-aligned");
              limitOffset = bitsToCharUnits(getFieldBitOffset(*probe));
              goto FoundLimit;
            }
          limitOffset = cxxRecordDecl ? astRecordLayout.getNonVirtualSize()
                                      : astRecordLayout.getDataSize();

        FoundLimit:
          CharUnits typeSize = getSize(type);
          if (beginOffset + typeSize <= limitOffset) {
            // There is space before limitOffset to create a naturally-sized
            // access unit.
            bestEndOffset = beginOffset + typeSize;
            bestEnd = field;
            bestClipped = false;
          }
          if (barrier) {
            // The next field is a barrier that we cannot merge across.
            installBest = true;
          } else if (cirGenTypes.getCGModule()
                         .getCodeGenOpts()
                         .FineGrainedBitfieldAccesses) {
            installBest = true;
          } else {
            // Otherwise, we're not installing. Update the bit size
            // of the current span to go all the way to limitOffset, which is
            // the (aligned) offset of next bitfield to consider.
            bitSizeSinceBegin = astContext.toBits(limitOffset - beginOffset);
          }
        }
      }
    }

    if (installBest) {
      assert((field == fieldEnd || !field->isBitField() ||
              (getFieldBitOffset(*field) % charBits) == 0) &&
             "Installing but not at an aligned bitfield or limit");
      CharUnits accessSize = bestEndOffset - beginOffset;
      if (!accessSize.isZero()) {
        // Add the storage member for the access unit to the record. The
        // bitfields get the offset of their storage but come afterward and
        // remain there after a stable sort.
        mlir::Type type;
        if (bestClipped) {
          assert(getSize(getUIntNType(astContext.toBits(accessSize))) >
                     accessSize &&
                 "Clipped access need not be clipped");
          type = getByteArrayType(accessSize);
        } else {
          type = getUIntNType(astContext.toBits(accessSize));
          assert(getSize(type) == accessSize &&
                 "Unclipped access must be clipped");
        }
        members.push_back(makeStorageInfo(beginOffset, type));
        for (; begin != bestEnd; ++begin)
          if (!begin->isZeroLengthBitField())
            members.push_back(MemberInfo(
                beginOffset, MemberInfo::InfoKind::Field, nullptr, *begin));
      }
      // Reset to start a new span.
      field = bestEnd;
      begin = fieldEnd;
    } else {
      assert(field != fieldEnd && field->isBitField() &&
             "Accumulating past end of bitfields");
      assert(!barrier && "Accumulating across barrier");
      // Accumulate this bitfield into the current (potential) span.
      bitSizeSinceBegin += field->getBitWidthValue();
      ++field;
    }
  }

  return field;
}

void CIRRecordLowering::accumulateFields() {
  for (RecordDecl::field_iterator field = recordDecl->field_begin(),
                                  fieldEnd = recordDecl->field_end();
       field != fieldEnd;) {
    if (field->isBitField()) {
      field = accumulateBitFields(field, fieldEnd);
      assert((field == fieldEnd || !field->isBitField()) &&
             "Failed to accumulate all the bitfields");
    } else if (!field->isZeroSize(astContext)) {
      members.push_back(MemberInfo(bitsToCharUnits(getFieldBitOffset(*field)),
                                   MemberInfo::InfoKind::Field,
                                   getStorageType(*field), *field));
      ++field;
    } else {
      // TODO(cir): do we want to do anything special about zero size members?
      assert(!cir::MissingFeatures::zeroSizeRecordMembers());
      ++field;
    }
  }
}

void CIRRecordLowering::calculateZeroInit() {
  for (const MemberInfo &member : members) {
    if (member.kind == MemberInfo::InfoKind::Field) {
      if (!member.fieldDecl || isZeroInitializable(member.fieldDecl))
        continue;
      zeroInitializable = zeroInitializableAsBase = false;
      return;
    } else if (member.kind == MemberInfo::InfoKind::Base ||
               member.kind == MemberInfo::InfoKind::VBase) {
      if (isZeroInitializable(member.cxxRecordDecl))
        continue;
      zeroInitializable = false;
      if (member.kind == MemberInfo::InfoKind::Base)
        zeroInitializableAsBase = false;
    }
  }
}

void CIRRecordLowering::determinePacked(bool nvBaseType) {
  if (packed)
    return;
  CharUnits alignment = CharUnits::One();
  CharUnits nvAlignment = CharUnits::One();
  CharUnits nvSize = !nvBaseType && cxxRecordDecl
                         ? astRecordLayout.getNonVirtualSize()
                         : CharUnits::Zero();

  for (const MemberInfo &member : members) {
    if (!member.data)
      continue;
    // If any member falls at an offset that it not a multiple of its alignment,
    // then the entire record must be packed.
    if (member.offset % getAlignment(member.data))
      packed = true;
    if (member.offset < nvSize)
      nvAlignment = std::max(nvAlignment, getAlignment(member.data));
    alignment = std::max(alignment, getAlignment(member.data));
  }
  // If the size of the record (the capstone's offset) is not a multiple of the
  // record's alignment, it must be packed.
  if (members.back().offset % alignment)
    packed = true;
  // If the non-virtual sub-object is not a multiple of the non-virtual
  // sub-object's alignment, it must be packed.  We cannot have a packed
  // non-virtual sub-object and an unpacked complete object or vise versa.
  if (nvSize % nvAlignment)
    packed = true;
  // Update the alignment of the sentinel.
  if (!packed)
    members.back().data = getUIntNType(astContext.toBits(alignment));
}

void CIRRecordLowering::insertPadding() {
  std::vector<std::pair<CharUnits, CharUnits>> padding;
  CharUnits size = CharUnits::Zero();
  for (const MemberInfo &member : members) {
    if (!member.data)
      continue;
    CharUnits offset = member.offset;
    assert(offset >= size);
    // Insert padding if we need to.
    if (offset !=
        size.alignTo(packed ? CharUnits::One() : getAlignment(member.data)))
      padding.push_back(std::make_pair(size, offset - size));
    size = offset + getSize(member.data);
  }
  if (padding.empty())
    return;
  padded = true;
  // Add the padding to the Members list and sort it.
  for (const std::pair<CharUnits, CharUnits> &paddingPair : padding)
    members.push_back(makeStorageInfo(paddingPair.first,
                                      getByteArrayType(paddingPair.second)));
  llvm::stable_sort(members);
}

std::unique_ptr<CIRGenRecordLayout>
CIRGenTypes::computeRecordLayout(const RecordDecl *rd, cir::RecordType *ty) {
  CIRRecordLowering lowering(*this, rd, /*packed=*/false);
  assert(ty->isIncomplete() && "recomputing record layout?");
  lowering.lower(/*nonVirtualBaseType=*/false);

  // If we're in C++, compute the base subobject type.
  cir::RecordType baseTy;
  if (llvm::isa<CXXRecordDecl>(rd) && !rd->isUnion() &&
      !rd->hasAttr<FinalAttr>()) {
    baseTy = *ty;
    if (lowering.astRecordLayout.getNonVirtualSize() !=
        lowering.astRecordLayout.getSize()) {
      CIRRecordLowering baseLowering(*this, rd, /*Packed=*/lowering.packed);
      baseLowering.lower(/*NonVirtualBaseType=*/true);
      std::string baseIdentifier = getRecordTypeName(rd, ".base");
      baseTy =
          builder.getCompleteRecordTy(baseLowering.fieldTypes, baseIdentifier,
                                      baseLowering.packed, baseLowering.padded);
      // TODO(cir): add something like addRecordTypeName

      // BaseTy and Ty must agree on their packedness for getCIRFieldNo to work
      // on both of them with the same index.
      assert(lowering.packed == baseLowering.packed &&
             "Non-virtual and complete types must agree on packedness");
    }
  }

  // Fill in the record *after* computing the base type.  Filling in the body
  // signifies that the type is no longer opaque and record layout is complete,
  // but we may need to recursively layout rd while laying D out as a base type.
  assert(!cir::MissingFeatures::astRecordDeclAttr());
  ty->complete(lowering.fieldTypes, lowering.packed, lowering.padded);

  auto rl = std::make_unique<CIRGenRecordLayout>(
      ty ? *ty : cir::RecordType{}, baseTy ? baseTy : cir::RecordType{},
      (bool)lowering.zeroInitializable, (bool)lowering.zeroInitializableAsBase);

  assert(!cir::MissingFeatures::recordZeroInit());

  rl->nonVirtualBases.swap(lowering.nonVirtualBases);
  rl->completeObjectVirtualBases.swap(lowering.virtualBases);

  assert(!cir::MissingFeatures::bitfields());

  // Add all the field numbers.
  rl->fieldIdxMap.swap(lowering.fieldIdxMap);

  rl->bitFields.swap(lowering.bitFields);

  // Dump the layout, if requested.
  if (getASTContext().getLangOpts().DumpRecordLayouts) {
    llvm::outs() << "\n*** Dumping CIRgen Record Layout\n";
    llvm::outs() << "Record: ";
    rd->dump(llvm::outs());
    llvm::outs() << "\nLayout: ";
    rl->print(llvm::outs());
  }

  // TODO: implement verification
  return rl;
}

void CIRGenRecordLayout::print(raw_ostream &os) const {
  os << "<CIRecordLayout\n";
  os << "   CIR Type:" << completeObjectType << "\n";
  if (baseSubobjectType)
    os << "   NonVirtualBaseCIRType:" << baseSubobjectType << "\n";
  os << "   IsZeroInitializable:" << zeroInitializable << "\n";
  os << "   BitFields:[\n";
  std::vector<std::pair<unsigned, const CIRGenBitFieldInfo *>> bitInfo;
  for (auto &[decl, info] : bitFields) {
    const RecordDecl *rd = decl->getParent();
    unsigned index = 0;
    for (RecordDecl::field_iterator it = rd->field_begin(); *it != decl; ++it)
      ++index;
    bitInfo.push_back(std::make_pair(index, &info));
  }
  llvm::array_pod_sort(bitInfo.begin(), bitInfo.end());
  for (std::pair<unsigned, const CIRGenBitFieldInfo *> &info : bitInfo) {
    os.indent(4);
    info.second->print(os);
    os << "\n";
  }
  os << "   ]>\n";
}

void CIRGenBitFieldInfo::print(raw_ostream &os) const {
  os << "<CIRBitFieldInfo" << " name:" << name << " offset:" << offset
     << " size:" << size << " isSigned:" << isSigned
     << " storageSize:" << storageSize
     << " storageOffset:" << storageOffset.getQuantity()
     << " volatileOffset:" << volatileOffset
     << " volatileStorageSize:" << volatileStorageSize
     << " volatileStorageOffset:" << volatileStorageOffset.getQuantity() << ">";
}

void CIRGenRecordLayout::dump() const { print(llvm::errs()); }

void CIRGenBitFieldInfo::dump() const { print(llvm::errs()); }

void CIRRecordLowering::lowerUnion() {
  CharUnits layoutSize = astRecordLayout.getSize();
  mlir::Type storageType = nullptr;
  bool seenNamedMember = false;

  // Iterate through the fields setting bitFieldInfo and the Fields array. Also
  // locate the "most appropriate" storage type.
  for (const FieldDecl *field : recordDecl->fields()) {
    mlir::Type fieldType;
    if (field->isBitField()) {
      if (field->isZeroLengthBitField())
        continue;
      fieldType = getBitfieldStorageType(field->getBitWidthValue());
      setBitFieldInfo(field, CharUnits::Zero(), fieldType);
    } else {
      fieldType = getStorageType(field);
    }

    // This maps a field to its index. For unions, the index is always 0.
    fieldIdxMap[field->getCanonicalDecl()] = 0;

    // Compute zero-initializable status.
    // This union might not be zero initialized: it may contain a pointer to
    // data member which might have some exotic initialization sequence.
    // If this is the case, then we ought not to try and come up with a "better"
    // type, it might not be very easy to come up with a Constant which
    // correctly initializes it.
    if (!seenNamedMember) {
      seenNamedMember = field->getIdentifier();
      if (!seenNamedMember)
        if (const RecordDecl *fieldRD = field->getType()->getAsRecordDecl())
          seenNamedMember = fieldRD->findFirstNamedDataMember();
      if (seenNamedMember && !isZeroInitializable(field)) {
        zeroInitializable = zeroInitializableAsBase = false;
        storageType = fieldType;
      }
    }

    // Because our union isn't zero initializable, we won't be getting a better
    // storage type.
    if (!zeroInitializable)
      continue;

    // Conditionally update our storage type if we've got a new "better" one.
    if (!storageType || getAlignment(fieldType) > getAlignment(storageType) ||
        (getAlignment(fieldType) == getAlignment(storageType) &&
         getSize(fieldType) > getSize(storageType)))
      storageType = fieldType;

    // NOTE(cir): Track all union member's types, not just the largest one. It
    // allows for proper type-checking and retain more info for analisys.
    fieldTypes.push_back(fieldType);
  }

  if (!storageType)
    cirGenTypes.getCGModule().errorNYI(recordDecl->getSourceRange(),
                                       "No-storage Union NYI");

  if (layoutSize < getSize(storageType))
    storageType = getByteArrayType(layoutSize);
  else
    appendPaddingBytes(layoutSize - getSize(storageType));

  // Set packed if we need it.
  if (layoutSize % getAlignment(storageType))
    packed = true;
}

bool CIRRecordLowering::hasOwnStorage(const CXXRecordDecl *decl,
                                      const CXXRecordDecl *query) {
  const ASTRecordLayout &declLayout = astContext.getASTRecordLayout(decl);
  if (declLayout.isPrimaryBaseVirtual() && declLayout.getPrimaryBase() == query)
    return false;
  for (const auto &base : decl->bases())
    if (!hasOwnStorage(base.getType()->getAsCXXRecordDecl(), query))
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
      !cirGenTypes.getCGModule().getCodeGenOpts().AAPCSBitfieldWidth)
    return;

  for (auto &[field, info] : bitFields) {
    mlir::Type resLTy = cirGenTypes.convertTypeForMem(field->getType());

    if (astContext.toBits(astRecordLayout.getAlignment()) <
        getSizeInBits(resLTy).getQuantity())
      continue;

    // CIRRecordLowering::setBitFieldInfo() pre-adjusts the bit-field offsets
    // for big-endian targets, but it assumes a container of width
    // info.storageSize. Since AAPCS uses a different container size (width
    // of the type), we first undo that calculation here and redo it once
    // the bit-field offset within the new container is calculated.
    const unsigned oldOffset =
        isBigEndian() ? info.storageSize - (info.offset + info.size)
                      : info.offset;
    // Offset to the bit-field from the beginning of the struct.
    const unsigned absoluteOffset =
        astContext.toBits(info.storageOffset) + oldOffset;

    // Container size is the width of the bit-field type.
    const unsigned storageSize = getSizeInBits(resLTy).getQuantity();
    // Nothing to do if the access uses the desired
    // container width and is naturally aligned.
    if (info.storageSize == storageSize && (oldOffset % storageSize == 0))
      continue;

    // Offset within the container.
    unsigned offset = absoluteOffset & (storageSize - 1);
    // Bail out if an aligned load of the container cannot cover the entire
    // bit-field. This can happen for example, if the bit-field is part of a
    // packed struct. AAPCS does not define access rules for such cases, we let
    // clang to follow its own rules.
    if (offset + info.size > storageSize)
      continue;

    // Re-adjust offsets for big-endian targets.
    if (isBigEndian())
      offset = storageSize - (offset + info.size);

    const CharUnits storageOffset =
        astContext.toCharUnitsFromBits(absoluteOffset & ~(storageSize - 1));
    const CharUnits end = storageOffset +
                          astContext.toCharUnitsFromBits(storageSize) -
                          CharUnits::One();

    const ASTRecordLayout &layout =
        astContext.getASTRecordLayout(field->getParent());
    // If we access outside memory outside the record, than bail out.
    const CharUnits recordSize = layout.getSize();
    if (end >= recordSize)
      continue;

    // Bail out if performing this load would access non-bit-fields members.
    bool conflict = false;
    for (const auto *f : recordDecl->fields()) {
      // Allow sized bit-fields overlaps.
      if (f->isBitField() && !f->isZeroLengthBitField())
        continue;

      const CharUnits fOffset = astContext.toCharUnitsFromBits(
          layout.getFieldOffset(f->getFieldIndex()));

      // As C11 defines, a zero sized bit-field defines a barrier, so
      // fields after and before it should be race condition free.
      // The AAPCS acknowledges it and imposes no restritions when the
      // natural container overlaps a zero-length bit-field.
      if (f->isZeroLengthBitField()) {
        if (end > fOffset && storageOffset < fOffset) {
          conflict = true;
          break;
        }
      }

      const CharUnits fEnd =
          fOffset +
          astContext.toCharUnitsFromBits(
              getSizeInBits(cirGenTypes.convertTypeForMem(f->getType()))
                  .getQuantity()) -
          CharUnits::One();
      // If no overlap, continue.
      if (end < fOffset || fEnd < storageOffset)
        continue;

      // The desired load overlaps a non-bit-field member, bail out.
      conflict = true;
      break;
    }

    if (conflict)
      continue;
    // Write the new bit-field access parameters.
    // As the storage offset now is defined as the number of elements from the
    // start of the structure, we should divide the Offset by the element size.
    info.volatileStorageOffset =
        storageOffset /
        astContext.toCharUnitsFromBits(storageSize).getQuantity();
    info.volatileStorageSize = storageSize;
    info.volatileOffset = offset;
  }
}

void CIRRecordLowering::accumulateBases() {
  // If we've got a primary virtual base, we need to add it with the bases.
  if (astRecordLayout.isPrimaryBaseVirtual()) {
    cirGenTypes.getCGModule().errorNYI(recordDecl->getSourceRange(),
                                       "accumulateBases: primary virtual base");
  }

  // Accumulate the non-virtual bases.
  for (const auto &base : cxxRecordDecl->bases()) {
    if (base.isVirtual())
      continue;
    // Bases can be zero-sized even if not technically empty if they
    // contain only a trailing array member.
    const CXXRecordDecl *baseDecl = base.getType()->getAsCXXRecordDecl();
    if (!baseDecl->isEmpty() &&
        !astContext.getASTRecordLayout(baseDecl).getNonVirtualSize().isZero()) {
      members.push_back(MemberInfo(astRecordLayout.getBaseClassOffset(baseDecl),
                                   MemberInfo::InfoKind::Base,
                                   getStorageType(baseDecl), baseDecl));
    }
  }
}

void CIRRecordLowering::accumulateVBases() {
  for (const auto &base : cxxRecordDecl->vbases()) {
    const CXXRecordDecl *baseDecl = base.getType()->getAsCXXRecordDecl();
    if (isEmptyRecordForLayout(astContext, base.getType()))
      continue;
    CharUnits offset = astRecordLayout.getVBaseClassOffset(baseDecl);
    // If the vbase is a primary virtual base of some base, then it doesn't
    // get its own storage location but instead lives inside of that base.
    if (isOverlappingVBaseABI() && astContext.isNearlyEmpty(baseDecl) &&
        !hasOwnStorage(cxxRecordDecl, baseDecl)) {
      members.push_back(
          MemberInfo(offset, MemberInfo::InfoKind::VBase, nullptr, baseDecl));
      continue;
    }
    // If we've got a vtordisp, add it as a storage type.
    if (astRecordLayout.getVBaseOffsetsMap()
            .find(baseDecl)
            ->second.hasVtorDisp())
      members.push_back(makeStorageInfo(offset - CharUnits::fromQuantity(4),
                                        getUIntNType(32)));
    members.push_back(MemberInfo(offset, MemberInfo::InfoKind::VBase,
                                 getStorageType(baseDecl), baseDecl));
  }
}

void CIRRecordLowering::accumulateVPtrs() {
  if (astRecordLayout.hasOwnVFPtr())
    members.push_back(MemberInfo(CharUnits::Zero(), MemberInfo::InfoKind::VFPtr,
                                 getVFPtrType()));

  if (astRecordLayout.hasOwnVBPtr())
    cirGenTypes.getCGModule().errorNYI(recordDecl->getSourceRange(),
                                       "accumulateVPtrs: hasOwnVBPtr");
}

mlir::Type CIRRecordLowering::getVFPtrType() {
  return cir::VPtrType::get(builder.getContext());
}
