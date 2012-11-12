//===--- CGRecordLayoutBuilder.cpp - CGRecordLayout builder  ----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Builder implementation for CGRecordLayout objects.
//
//===----------------------------------------------------------------------===//

#include "CGRecordLayout.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/Attr.h"
#include "clang/AST/CXXInheritance.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/Expr.h"
#include "clang/AST/RecordLayout.h"
#include "clang/Frontend/CodeGenOptions.h"
#include "CodeGenTypes.h"
#include "CGCXXABI.h"
#include "llvm/DerivedTypes.h"
#include "llvm/Type.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/DataLayout.h"
using namespace clang;
using namespace CodeGen;

namespace {

class CGRecordLayoutBuilder {
public:
  /// FieldTypes - Holds the LLVM types that the struct is created from.
  /// 
  SmallVector<llvm::Type *, 16> FieldTypes;

  /// BaseSubobjectType - Holds the LLVM type for the non-virtual part
  /// of the struct. For example, consider:
  ///
  /// struct A { int i; };
  /// struct B { void *v; };
  /// struct C : virtual A, B { };
  ///
  /// The LLVM type of C will be
  /// %struct.C = type { i32 (...)**, %struct.A, i32, %struct.B }
  ///
  /// And the LLVM type of the non-virtual base struct will be
  /// %struct.C.base = type { i32 (...)**, %struct.A, i32 }
  ///
  /// This only gets initialized if the base subobject type is
  /// different from the complete-object type.
  llvm::StructType *BaseSubobjectType;

  /// FieldInfo - Holds a field and its corresponding LLVM field number.
  llvm::DenseMap<const FieldDecl *, unsigned> Fields;

  /// BitFieldInfo - Holds location and size information about a bit field.
  llvm::DenseMap<const FieldDecl *, CGBitFieldInfo> BitFields;

  llvm::DenseMap<const CXXRecordDecl *, unsigned> NonVirtualBases;
  llvm::DenseMap<const CXXRecordDecl *, unsigned> VirtualBases;

  /// IndirectPrimaryBases - Virtual base classes, direct or indirect, that are
  /// primary base classes for some other direct or indirect base class.
  CXXIndirectPrimaryBaseSet IndirectPrimaryBases;

  /// LaidOutVirtualBases - A set of all laid out virtual bases, used to avoid
  /// avoid laying out virtual bases more than once.
  llvm::SmallPtrSet<const CXXRecordDecl *, 4> LaidOutVirtualBases;
  
  /// IsZeroInitializable - Whether this struct can be C++
  /// zero-initialized with an LLVM zeroinitializer.
  bool IsZeroInitializable;
  bool IsZeroInitializableAsBase;

  /// Packed - Whether the resulting LLVM struct will be packed or not.
  bool Packed;
  
  /// IsMsStruct - Whether ms_struct is in effect or not
  bool IsMsStruct;

private:
  CodeGenTypes &Types;

  /// LastLaidOutBaseInfo - Contains the offset and non-virtual size of the
  /// last base laid out. Used so that we can replace the last laid out base
  /// type with an i8 array if needed.
  struct LastLaidOutBaseInfo {
    CharUnits Offset;
    CharUnits NonVirtualSize;

    bool isValid() const { return !NonVirtualSize.isZero(); }
    void invalidate() { NonVirtualSize = CharUnits::Zero(); }
  
  } LastLaidOutBase;

  /// Alignment - Contains the alignment of the RecordDecl.
  CharUnits Alignment;

  /// BitsAvailableInLastField - If a bit field spans only part of a LLVM field,
  /// this will have the number of bits still available in the field.
  char BitsAvailableInLastField;

  /// NextFieldOffset - Holds the next field offset.
  CharUnits NextFieldOffset;

  /// LayoutUnionField - Will layout a field in an union and return the type
  /// that the field will have.
  llvm::Type *LayoutUnionField(const FieldDecl *Field,
                               const ASTRecordLayout &Layout);
  
  /// LayoutUnion - Will layout a union RecordDecl.
  void LayoutUnion(const RecordDecl *D);

  /// LayoutField - try to layout all fields in the record decl.
  /// Returns false if the operation failed because the struct is not packed.
  bool LayoutFields(const RecordDecl *D);

  /// Layout a single base, virtual or non-virtual
  bool LayoutBase(const CXXRecordDecl *base,
                  const CGRecordLayout &baseLayout,
                  CharUnits baseOffset);

  /// LayoutVirtualBase - layout a single virtual base.
  bool LayoutVirtualBase(const CXXRecordDecl *base,
                         CharUnits baseOffset);

  /// LayoutVirtualBases - layout the virtual bases of a record decl.
  bool LayoutVirtualBases(const CXXRecordDecl *RD,
                          const ASTRecordLayout &Layout);

  /// MSLayoutVirtualBases - layout the virtual bases of a record decl,
  /// like MSVC.
  bool MSLayoutVirtualBases(const CXXRecordDecl *RD,
                            const ASTRecordLayout &Layout);
  
  /// LayoutNonVirtualBase - layout a single non-virtual base.
  bool LayoutNonVirtualBase(const CXXRecordDecl *base,
                            CharUnits baseOffset);
  
  /// LayoutNonVirtualBases - layout the virtual bases of a record decl.
  bool LayoutNonVirtualBases(const CXXRecordDecl *RD, 
                             const ASTRecordLayout &Layout);

  /// ComputeNonVirtualBaseType - Compute the non-virtual base field types.
  bool ComputeNonVirtualBaseType(const CXXRecordDecl *RD);
  
  /// LayoutField - layout a single field. Returns false if the operation failed
  /// because the current struct is not packed.
  bool LayoutField(const FieldDecl *D, uint64_t FieldOffset);

  /// LayoutBitField - layout a single bit field.
  void LayoutBitField(const FieldDecl *D, uint64_t FieldOffset);

  /// AppendField - Appends a field with the given offset and type.
  void AppendField(CharUnits fieldOffset, llvm::Type *FieldTy);

  /// AppendPadding - Appends enough padding bytes so that the total
  /// struct size is a multiple of the field alignment.
  void AppendPadding(CharUnits fieldOffset, CharUnits fieldAlignment);

  /// ResizeLastBaseFieldIfNecessary - Fields and bases can be laid out in the
  /// tail padding of a previous base. If this happens, the type of the previous
  /// base needs to be changed to an array of i8. Returns true if the last
  /// laid out base was resized.
  bool ResizeLastBaseFieldIfNecessary(CharUnits offset);

  /// getByteArrayType - Returns a byte array type with the given number of
  /// elements.
  llvm::Type *getByteArrayType(CharUnits NumBytes);
  
  /// AppendBytes - Append a given number of bytes to the record.
  void AppendBytes(CharUnits numBytes);

  /// AppendTailPadding - Append enough tail padding so that the type will have
  /// the passed size.
  void AppendTailPadding(CharUnits RecordSize);

  CharUnits getTypeAlignment(llvm::Type *Ty) const;

  /// getAlignmentAsLLVMStruct - Returns the maximum alignment of all the
  /// LLVM element types.
  CharUnits getAlignmentAsLLVMStruct() const;

  /// CheckZeroInitializable - Check if the given type contains a pointer
  /// to data member.
  void CheckZeroInitializable(QualType T);

public:
  CGRecordLayoutBuilder(CodeGenTypes &Types)
    : BaseSubobjectType(0),
      IsZeroInitializable(true), IsZeroInitializableAsBase(true),
      Packed(false), IsMsStruct(false),
      Types(Types), BitsAvailableInLastField(0) { }

  /// Layout - Will layout a RecordDecl.
  void Layout(const RecordDecl *D);
};

}

void CGRecordLayoutBuilder::Layout(const RecordDecl *D) {
  Alignment = Types.getContext().getASTRecordLayout(D).getAlignment();
  Packed = D->hasAttr<PackedAttr>();
  
  IsMsStruct = D->isMsStruct(Types.getContext());

  if (D->isUnion()) {
    LayoutUnion(D);
    return;
  }

  if (LayoutFields(D))
    return;

  // We weren't able to layout the struct. Try again with a packed struct
  Packed = true;
  LastLaidOutBase.invalidate();
  NextFieldOffset = CharUnits::Zero();
  FieldTypes.clear();
  Fields.clear();
  BitFields.clear();
  NonVirtualBases.clear();
  VirtualBases.clear();

  LayoutFields(D);
}

CGBitFieldInfo CGBitFieldInfo::MakeInfo(CodeGenTypes &Types,
                               const FieldDecl *FD,
                               uint64_t FieldOffset,
                               uint64_t FieldSize,
                               uint64_t ContainingTypeSizeInBits,
                               unsigned ContainingTypeAlign) {
  assert(ContainingTypeAlign && "Expected alignment to be specified");

  llvm::Type *Ty = Types.ConvertTypeForMem(FD->getType());
  CharUnits TypeSizeInBytes =
    CharUnits::fromQuantity(Types.getDataLayout().getTypeAllocSize(Ty));
  uint64_t TypeSizeInBits = Types.getContext().toBits(TypeSizeInBytes);

  bool IsSigned = FD->getType()->isSignedIntegerOrEnumerationType();

  if (FieldSize > TypeSizeInBits) {
    // We have a wide bit-field. The extra bits are only used for padding, so
    // if we have a bitfield of type T, with size N:
    //
    // T t : N;
    //
    // We can just assume that it's:
    //
    // T t : sizeof(T);
    //
    FieldSize = TypeSizeInBits;
  }

  // in big-endian machines the first fields are in higher bit positions,
  // so revert the offset. The byte offsets are reversed(back) later.
  if (Types.getDataLayout().isBigEndian()) {
    FieldOffset = ((ContainingTypeSizeInBits)-FieldOffset-FieldSize);
  }

  // Compute the access components. The policy we use is to start by attempting
  // to access using the width of the bit-field type itself and to always access
  // at aligned indices of that type. If such an access would fail because it
  // extends past the bound of the type, then we reduce size to the next smaller
  // power of two and retry. The current algorithm assumes pow2 sized types,
  // although this is easy to fix.
  //
  assert(llvm::isPowerOf2_32(TypeSizeInBits) && "Unexpected type size!");
  CGBitFieldInfo::AccessInfo Components[3];
  unsigned NumComponents = 0;
  unsigned AccessedTargetBits = 0;       // The number of target bits accessed.
  unsigned AccessWidth = TypeSizeInBits; // The current access width to attempt.

  // If requested, widen the initial bit-field access to be register sized. The
  // theory is that this is most likely to allow multiple accesses into the same
  // structure to be coalesced, and that the backend should be smart enough to
  // narrow the store if no coalescing is ever done.
  //
  // The subsequent code will handle align these access to common boundaries and
  // guaranteeing that we do not access past the end of the structure.
  if (Types.getCodeGenOpts().UseRegisterSizedBitfieldAccess) {
    if (AccessWidth < Types.getTarget().getRegisterWidth())
      AccessWidth = Types.getTarget().getRegisterWidth();
  }

  // Round down from the field offset to find the first access position that is
  // at an aligned offset of the initial access type.
  uint64_t AccessStart = FieldOffset - (FieldOffset % AccessWidth);

  // Adjust initial access size to fit within record.
  while (AccessWidth > Types.getTarget().getCharWidth() &&
         AccessStart + AccessWidth > ContainingTypeSizeInBits) {
    AccessWidth >>= 1;
    AccessStart = FieldOffset - (FieldOffset % AccessWidth);
  }

  while (AccessedTargetBits < FieldSize) {
    // Check that we can access using a type of this size, without reading off
    // the end of the structure. This can occur with packed structures and
    // -fno-bitfield-type-align, for example.
    if (AccessStart + AccessWidth > ContainingTypeSizeInBits) {
      // If so, reduce access size to the next smaller power-of-two and retry.
      AccessWidth >>= 1;
      assert(AccessWidth >= Types.getTarget().getCharWidth()
             && "Cannot access under byte size!");
      continue;
    }

    // Otherwise, add an access component.

    // First, compute the bits inside this access which are part of the
    // target. We are reading bits [AccessStart, AccessStart + AccessWidth); the
    // intersection with [FieldOffset, FieldOffset + FieldSize) gives the bits
    // in the target that we are reading.
    assert(FieldOffset < AccessStart + AccessWidth && "Invalid access start!");
    assert(AccessStart < FieldOffset + FieldSize && "Invalid access start!");
    uint64_t AccessBitsInFieldStart = std::max(AccessStart, FieldOffset);
    uint64_t AccessBitsInFieldSize =
      std::min(AccessWidth + AccessStart,
               FieldOffset + FieldSize) - AccessBitsInFieldStart;

    assert(NumComponents < 3 && "Unexpected number of components!");
    CGBitFieldInfo::AccessInfo &AI = Components[NumComponents++];
    AI.FieldIndex = 0;
    // FIXME: We still follow the old access pattern of only using the field
    // byte offset. We should switch this once we fix the struct layout to be
    // pretty.

    // on big-endian machines we reverted the bit offset because first fields are
    // in higher bits. But this also reverts the bytes, so fix this here by reverting
    // the byte offset on big-endian machines.
    if (Types.getDataLayout().isBigEndian()) {
      AI.FieldByteOffset = Types.getContext().toCharUnitsFromBits(
          ContainingTypeSizeInBits - AccessStart - AccessWidth);
    } else {
      AI.FieldByteOffset = Types.getContext().toCharUnitsFromBits(AccessStart);
    }
    AI.FieldBitStart = AccessBitsInFieldStart - AccessStart;
    AI.AccessWidth = AccessWidth;
    AI.AccessAlignment = Types.getContext().toCharUnitsFromBits(
        llvm::MinAlign(ContainingTypeAlign, AccessStart));
    AI.TargetBitOffset = AccessedTargetBits;
    AI.TargetBitWidth = AccessBitsInFieldSize;

    AccessStart += AccessWidth;
    AccessedTargetBits += AI.TargetBitWidth;
  }

  assert(AccessedTargetBits == FieldSize && "Invalid bit-field access!");
  return CGBitFieldInfo(FieldSize, NumComponents, Components, IsSigned);
}

CGBitFieldInfo CGBitFieldInfo::MakeInfo(CodeGenTypes &Types,
                                        const FieldDecl *FD,
                                        uint64_t FieldOffset,
                                        uint64_t FieldSize) {
  const RecordDecl *RD = FD->getParent();
  const ASTRecordLayout &RL = Types.getContext().getASTRecordLayout(RD);
  uint64_t ContainingTypeSizeInBits = Types.getContext().toBits(RL.getSize());
  unsigned ContainingTypeAlign = Types.getContext().toBits(RL.getAlignment());

  return MakeInfo(Types, FD, FieldOffset, FieldSize, ContainingTypeSizeInBits,
                  ContainingTypeAlign);
}

void CGRecordLayoutBuilder::LayoutBitField(const FieldDecl *D,
                                           uint64_t fieldOffset) {
  uint64_t fieldSize = D->getBitWidthValue(Types.getContext());

  if (fieldSize == 0)
    return;

  uint64_t nextFieldOffsetInBits = Types.getContext().toBits(NextFieldOffset);
  CharUnits numBytesToAppend;
  unsigned charAlign = Types.getContext().getTargetInfo().getCharAlign();

  if (fieldOffset < nextFieldOffsetInBits && !BitsAvailableInLastField) {
    assert(fieldOffset % charAlign == 0 && 
           "Field offset not aligned correctly");

    CharUnits fieldOffsetInCharUnits = 
      Types.getContext().toCharUnitsFromBits(fieldOffset);

    // Try to resize the last base field.
    if (ResizeLastBaseFieldIfNecessary(fieldOffsetInCharUnits))
      nextFieldOffsetInBits = Types.getContext().toBits(NextFieldOffset);
  }

  if (fieldOffset < nextFieldOffsetInBits) {
    assert(BitsAvailableInLastField && "Bitfield size mismatch!");
    assert(!NextFieldOffset.isZero() && "Must have laid out at least one byte");

    // The bitfield begins in the previous bit-field.
    numBytesToAppend = Types.getContext().toCharUnitsFromBits(
      llvm::RoundUpToAlignment(fieldSize - BitsAvailableInLastField, 
                               charAlign));
  } else {
    assert(fieldOffset % charAlign == 0 && 
           "Field offset not aligned correctly");

    // Append padding if necessary.
    AppendPadding(Types.getContext().toCharUnitsFromBits(fieldOffset), 
                  CharUnits::One());

    numBytesToAppend = Types.getContext().toCharUnitsFromBits(
        llvm::RoundUpToAlignment(fieldSize, charAlign));

    assert(!numBytesToAppend.isZero() && "No bytes to append!");
  }

  // Add the bit field info.
  BitFields.insert(std::make_pair(D,
                   CGBitFieldInfo::MakeInfo(Types, D, fieldOffset, fieldSize)));

  AppendBytes(numBytesToAppend);

  BitsAvailableInLastField =
    Types.getContext().toBits(NextFieldOffset) - (fieldOffset + fieldSize);
}

bool CGRecordLayoutBuilder::LayoutField(const FieldDecl *D,
                                        uint64_t fieldOffset) {
  // If the field is packed, then we need a packed struct.
  if (!Packed && D->hasAttr<PackedAttr>())
    return false;

  if (D->isBitField()) {
    // We must use packed structs for unnamed bit fields since they
    // don't affect the struct alignment.
    if (!Packed && !D->getDeclName())
      return false;

    LayoutBitField(D, fieldOffset);
    return true;
  }

  CheckZeroInitializable(D->getType());

  assert(fieldOffset % Types.getTarget().getCharWidth() == 0
         && "field offset is not on a byte boundary!");
  CharUnits fieldOffsetInBytes
    = Types.getContext().toCharUnitsFromBits(fieldOffset);

  llvm::Type *Ty = Types.ConvertTypeForMem(D->getType());
  CharUnits typeAlignment = getTypeAlignment(Ty);

  // If the type alignment is larger then the struct alignment, we must use
  // a packed struct.
  if (typeAlignment > Alignment) {
    assert(!Packed && "Alignment is wrong even with packed struct!");
    return false;
  }

  if (!Packed) {
    if (const RecordType *RT = D->getType()->getAs<RecordType>()) {
      const RecordDecl *RD = cast<RecordDecl>(RT->getDecl());
      if (const MaxFieldAlignmentAttr *MFAA =
            RD->getAttr<MaxFieldAlignmentAttr>()) {
        if (MFAA->getAlignment() != Types.getContext().toBits(typeAlignment))
          return false;
      }
    }
  }

  // Round up the field offset to the alignment of the field type.
  CharUnits alignedNextFieldOffsetInBytes =
    NextFieldOffset.RoundUpToAlignment(typeAlignment);

  if (fieldOffsetInBytes < alignedNextFieldOffsetInBytes) {
    // Try to resize the last base field.
    if (ResizeLastBaseFieldIfNecessary(fieldOffsetInBytes)) {
      alignedNextFieldOffsetInBytes = 
        NextFieldOffset.RoundUpToAlignment(typeAlignment);
    }
  }

  if (fieldOffsetInBytes < alignedNextFieldOffsetInBytes) {
    assert(!Packed && "Could not place field even with packed struct!");
    return false;
  }

  AppendPadding(fieldOffsetInBytes, typeAlignment);

  // Now append the field.
  Fields[D] = FieldTypes.size();
  AppendField(fieldOffsetInBytes, Ty);

  LastLaidOutBase.invalidate();
  return true;
}

llvm::Type *
CGRecordLayoutBuilder::LayoutUnionField(const FieldDecl *Field,
                                        const ASTRecordLayout &Layout) {
  if (Field->isBitField()) {
    uint64_t FieldSize = Field->getBitWidthValue(Types.getContext());

    // Ignore zero sized bit fields.
    if (FieldSize == 0)
      return 0;

    llvm::Type *FieldTy = llvm::Type::getInt8Ty(Types.getLLVMContext());
    CharUnits NumBytesToAppend = Types.getContext().toCharUnitsFromBits(
      llvm::RoundUpToAlignment(FieldSize, 
                               Types.getContext().getTargetInfo().getCharAlign()));

    if (NumBytesToAppend > CharUnits::One())
      FieldTy = llvm::ArrayType::get(FieldTy, NumBytesToAppend.getQuantity());

    // Add the bit field info.
    BitFields.insert(std::make_pair(Field,
                         CGBitFieldInfo::MakeInfo(Types, Field, 0, FieldSize)));
    return FieldTy;
  }

  // This is a regular union field.
  Fields[Field] = 0;
  return Types.ConvertTypeForMem(Field->getType());
}

void CGRecordLayoutBuilder::LayoutUnion(const RecordDecl *D) {
  assert(D->isUnion() && "Can't call LayoutUnion on a non-union record!");

  const ASTRecordLayout &layout = Types.getContext().getASTRecordLayout(D);

  llvm::Type *unionType = 0;
  CharUnits unionSize = CharUnits::Zero();
  CharUnits unionAlign = CharUnits::Zero();

  bool hasOnlyZeroSizedBitFields = true;
  bool checkedFirstFieldZeroInit = false;

  unsigned fieldNo = 0;
  for (RecordDecl::field_iterator field = D->field_begin(),
       fieldEnd = D->field_end(); field != fieldEnd; ++field, ++fieldNo) {
    assert(layout.getFieldOffset(fieldNo) == 0 &&
          "Union field offset did not start at the beginning of record!");
    llvm::Type *fieldType = LayoutUnionField(*field, layout);

    if (!fieldType)
      continue;

    if (field->getDeclName() && !checkedFirstFieldZeroInit) {
      CheckZeroInitializable(field->getType());
      checkedFirstFieldZeroInit = true;
    }

    hasOnlyZeroSizedBitFields = false;

    CharUnits fieldAlign = CharUnits::fromQuantity(
                          Types.getDataLayout().getABITypeAlignment(fieldType));
    CharUnits fieldSize = CharUnits::fromQuantity(
                             Types.getDataLayout().getTypeAllocSize(fieldType));

    if (fieldAlign < unionAlign)
      continue;

    if (fieldAlign > unionAlign || fieldSize > unionSize) {
      unionType = fieldType;
      unionAlign = fieldAlign;
      unionSize = fieldSize;
    }
  }

  // Now add our field.
  if (unionType) {
    AppendField(CharUnits::Zero(), unionType);

    if (getTypeAlignment(unionType) > layout.getAlignment()) {
      // We need a packed struct.
      Packed = true;
      unionAlign = CharUnits::One();
    }
  }
  if (unionAlign.isZero()) {
    (void)hasOnlyZeroSizedBitFields;
    assert(hasOnlyZeroSizedBitFields &&
           "0-align record did not have all zero-sized bit-fields!");
    unionAlign = CharUnits::One();
  }

  // Append tail padding.
  CharUnits recordSize = layout.getSize();
  if (recordSize > unionSize)
    AppendPadding(recordSize, unionAlign);
}

bool CGRecordLayoutBuilder::LayoutBase(const CXXRecordDecl *base,
                                       const CGRecordLayout &baseLayout,
                                       CharUnits baseOffset) {
  ResizeLastBaseFieldIfNecessary(baseOffset);

  AppendPadding(baseOffset, CharUnits::One());

  const ASTRecordLayout &baseASTLayout
    = Types.getContext().getASTRecordLayout(base);

  LastLaidOutBase.Offset = NextFieldOffset;
  LastLaidOutBase.NonVirtualSize = baseASTLayout.getNonVirtualSize();

  llvm::StructType *subobjectType = baseLayout.getBaseSubobjectLLVMType();
  if (getTypeAlignment(subobjectType) > Alignment)
    return false;

  AppendField(baseOffset, subobjectType);
  return true;
}

bool CGRecordLayoutBuilder::LayoutNonVirtualBase(const CXXRecordDecl *base,
                                                 CharUnits baseOffset) {
  // Ignore empty bases.
  if (base->isEmpty()) return true;

  const CGRecordLayout &baseLayout = Types.getCGRecordLayout(base);
  if (IsZeroInitializableAsBase) {
    assert(IsZeroInitializable &&
           "class zero-initializable as base but not as complete object");

    IsZeroInitializable = IsZeroInitializableAsBase =
      baseLayout.isZeroInitializableAsBase();
  }

  if (!LayoutBase(base, baseLayout, baseOffset))
    return false;
  NonVirtualBases[base] = (FieldTypes.size() - 1);
  return true;
}

bool
CGRecordLayoutBuilder::LayoutVirtualBase(const CXXRecordDecl *base,
                                         CharUnits baseOffset) {
  // Ignore empty bases.
  if (base->isEmpty()) return true;

  const CGRecordLayout &baseLayout = Types.getCGRecordLayout(base);
  if (IsZeroInitializable)
    IsZeroInitializable = baseLayout.isZeroInitializableAsBase();

  if (!LayoutBase(base, baseLayout, baseOffset))
    return false;
  VirtualBases[base] = (FieldTypes.size() - 1);
  return true;
}

bool
CGRecordLayoutBuilder::MSLayoutVirtualBases(const CXXRecordDecl *RD,
                                          const ASTRecordLayout &Layout) {
  if (!RD->getNumVBases())
    return true;

  // The vbases list is uniqued and ordered by a depth-first
  // traversal, which is what we need here.
  for (CXXRecordDecl::base_class_const_iterator I = RD->vbases_begin(),
        E = RD->vbases_end(); I != E; ++I) {

    const CXXRecordDecl *BaseDecl = 
      cast<CXXRecordDecl>(I->getType()->castAs<RecordType>()->getDecl());

    CharUnits vbaseOffset = Layout.getVBaseClassOffset(BaseDecl);
    if (!LayoutVirtualBase(BaseDecl, vbaseOffset))
      return false;
  }
  return true;
}

/// LayoutVirtualBases - layout the non-virtual bases of a record decl.
bool
CGRecordLayoutBuilder::LayoutVirtualBases(const CXXRecordDecl *RD,
                                          const ASTRecordLayout &Layout) {
  for (CXXRecordDecl::base_class_const_iterator I = RD->bases_begin(),
       E = RD->bases_end(); I != E; ++I) {
    const CXXRecordDecl *BaseDecl = 
      cast<CXXRecordDecl>(I->getType()->getAs<RecordType>()->getDecl());

    // We only want to lay out virtual bases that aren't indirect primary bases
    // of some other base.
    if (I->isVirtual() && !IndirectPrimaryBases.count(BaseDecl)) {
      // Only lay out the base once.
      if (!LaidOutVirtualBases.insert(BaseDecl))
        continue;

      CharUnits vbaseOffset = Layout.getVBaseClassOffset(BaseDecl);
      if (!LayoutVirtualBase(BaseDecl, vbaseOffset))
        return false;
    }

    if (!BaseDecl->getNumVBases()) {
      // This base isn't interesting since it doesn't have any virtual bases.
      continue;
    }
    
    if (!LayoutVirtualBases(BaseDecl, Layout))
      return false;
  }
  return true;
}

bool
CGRecordLayoutBuilder::LayoutNonVirtualBases(const CXXRecordDecl *RD,
                                             const ASTRecordLayout &Layout) {
  const CXXRecordDecl *PrimaryBase = Layout.getPrimaryBase();

  // If we have a primary base, lay it out first.
  if (PrimaryBase) {
    if (!Layout.isPrimaryBaseVirtual()) {
      if (!LayoutNonVirtualBase(PrimaryBase, CharUnits::Zero()))
        return false;
    } else {
      if (!LayoutVirtualBase(PrimaryBase, CharUnits::Zero()))
        return false;
    }

  // Otherwise, add a vtable / vf-table if the layout says to do so.
  } else if (Layout.hasOwnVFPtr()) {
    llvm::Type *FunctionType =
      llvm::FunctionType::get(llvm::Type::getInt32Ty(Types.getLLVMContext()),
                              /*isVarArg=*/true);
    llvm::Type *VTableTy = FunctionType->getPointerTo();

    if (getTypeAlignment(VTableTy) > Alignment) {
      // FIXME: Should we allow this to happen in Sema?
      assert(!Packed && "Alignment is wrong even with packed struct!");
      return false;
    }

    assert(NextFieldOffset.isZero() &&
           "VTable pointer must come first!");
    AppendField(CharUnits::Zero(), VTableTy->getPointerTo());
  }

  // Layout the non-virtual bases.
  for (CXXRecordDecl::base_class_const_iterator I = RD->bases_begin(),
       E = RD->bases_end(); I != E; ++I) {
    if (I->isVirtual())
      continue;

    const CXXRecordDecl *BaseDecl = 
      cast<CXXRecordDecl>(I->getType()->getAs<RecordType>()->getDecl());

    // We've already laid out the primary base.
    if (BaseDecl == PrimaryBase && !Layout.isPrimaryBaseVirtual())
      continue;

    if (!LayoutNonVirtualBase(BaseDecl, Layout.getBaseClassOffset(BaseDecl)))
      return false;
  }

  // Add a vb-table pointer if the layout insists.
  if (Layout.getVBPtrOffset() != CharUnits::fromQuantity(-1)) {
    CharUnits VBPtrOffset = Layout.getVBPtrOffset();
    llvm::Type *Vbptr = llvm::Type::getInt32PtrTy(Types.getLLVMContext());
    AppendPadding(VBPtrOffset, getTypeAlignment(Vbptr));
    AppendField(VBPtrOffset, Vbptr);
  }

  return true;
}

bool
CGRecordLayoutBuilder::ComputeNonVirtualBaseType(const CXXRecordDecl *RD) {
  const ASTRecordLayout &Layout = Types.getContext().getASTRecordLayout(RD);

  CharUnits NonVirtualSize  = Layout.getNonVirtualSize();
  CharUnits NonVirtualAlign = Layout.getNonVirtualAlign();
  CharUnits AlignedNonVirtualTypeSize =
    NonVirtualSize.RoundUpToAlignment(NonVirtualAlign);
  
  // First check if we can use the same fields as for the complete class.
  CharUnits RecordSize = Layout.getSize();
  if (AlignedNonVirtualTypeSize == RecordSize)
    return true;

  // Check if we need padding.
  CharUnits AlignedNextFieldOffset =
    NextFieldOffset.RoundUpToAlignment(getAlignmentAsLLVMStruct());

  if (AlignedNextFieldOffset > AlignedNonVirtualTypeSize) {
    assert(!Packed && "cannot layout even as packed struct");
    return false; // Needs packing.
  }

  bool needsPadding = (AlignedNonVirtualTypeSize != AlignedNextFieldOffset);
  if (needsPadding) {
    CharUnits NumBytes = AlignedNonVirtualTypeSize - AlignedNextFieldOffset;
    FieldTypes.push_back(getByteArrayType(NumBytes));
  }
  
  BaseSubobjectType = llvm::StructType::create(Types.getLLVMContext(),
                                               FieldTypes, "", Packed);
  Types.addRecordTypeName(RD, BaseSubobjectType, ".base");

  // Pull the padding back off.
  if (needsPadding)
    FieldTypes.pop_back();

  return true;
}

bool CGRecordLayoutBuilder::LayoutFields(const RecordDecl *D) {
  assert(!D->isUnion() && "Can't call LayoutFields on a union!");
  assert(!Alignment.isZero() && "Did not set alignment!");

  const ASTRecordLayout &Layout = Types.getContext().getASTRecordLayout(D);

  const CXXRecordDecl *RD = dyn_cast<CXXRecordDecl>(D);
  if (RD)
    if (!LayoutNonVirtualBases(RD, Layout))
      return false;

  unsigned FieldNo = 0;
  const FieldDecl *LastFD = 0;
  
  for (RecordDecl::field_iterator Field = D->field_begin(),
       FieldEnd = D->field_end(); Field != FieldEnd; ++Field, ++FieldNo) {
    if (IsMsStruct) {
      // Zero-length bitfields following non-bitfield members are
      // ignored:
      const FieldDecl *FD = *Field;
      if (Types.getContext().ZeroBitfieldFollowsNonBitfield(FD, LastFD)) {
        --FieldNo;
        continue;
      }
      LastFD = FD;
    }
    
    if (!LayoutField(*Field, Layout.getFieldOffset(FieldNo))) {
      assert(!Packed &&
             "Could not layout fields even with a packed LLVM struct!");
      return false;
    }
  }

  if (RD) {
    // We've laid out the non-virtual bases and the fields, now compute the
    // non-virtual base field types.
    if (!ComputeNonVirtualBaseType(RD)) {
      assert(!Packed && "Could not layout even with a packed LLVM struct!");
      return false;
    }

    // Lay out the virtual bases.  The MS ABI uses a different
    // algorithm here due to the lack of primary virtual bases.
    if (Types.getContext().getTargetInfo().getCXXABI() != CXXABI_Microsoft) {
      RD->getIndirectPrimaryBases(IndirectPrimaryBases);
      if (Layout.isPrimaryBaseVirtual())
        IndirectPrimaryBases.insert(Layout.getPrimaryBase());

      if (!LayoutVirtualBases(RD, Layout))
        return false;
    } else {
      if (!MSLayoutVirtualBases(RD, Layout))
        return false;
    }
  }
  
  // Append tail padding if necessary.
  AppendTailPadding(Layout.getSize());

  return true;
}

void CGRecordLayoutBuilder::AppendTailPadding(CharUnits RecordSize) {
  ResizeLastBaseFieldIfNecessary(RecordSize);

  assert(NextFieldOffset <= RecordSize && "Size mismatch!");

  CharUnits AlignedNextFieldOffset =
    NextFieldOffset.RoundUpToAlignment(getAlignmentAsLLVMStruct());

  if (AlignedNextFieldOffset == RecordSize) {
    // We don't need any padding.
    return;
  }

  CharUnits NumPadBytes = RecordSize - NextFieldOffset;
  AppendBytes(NumPadBytes);
}

void CGRecordLayoutBuilder::AppendField(CharUnits fieldOffset,
                                        llvm::Type *fieldType) {
  CharUnits fieldSize =
    CharUnits::fromQuantity(Types.getDataLayout().getTypeAllocSize(fieldType));

  FieldTypes.push_back(fieldType);

  NextFieldOffset = fieldOffset + fieldSize;
  BitsAvailableInLastField = 0;
}

void CGRecordLayoutBuilder::AppendPadding(CharUnits fieldOffset,
                                          CharUnits fieldAlignment) {
  assert(NextFieldOffset <= fieldOffset &&
         "Incorrect field layout!");

  // Do nothing if we're already at the right offset.
  if (fieldOffset == NextFieldOffset) return;

  // If we're not emitting a packed LLVM type, try to avoid adding
  // unnecessary padding fields.
  if (!Packed) {
    // Round up the field offset to the alignment of the field type.
    CharUnits alignedNextFieldOffset =
      NextFieldOffset.RoundUpToAlignment(fieldAlignment);
    assert(alignedNextFieldOffset <= fieldOffset);

    // If that's the right offset, we're done.
    if (alignedNextFieldOffset == fieldOffset) return;
  }

  // Otherwise we need explicit padding.
  CharUnits padding = fieldOffset - NextFieldOffset;
  AppendBytes(padding);
}

bool CGRecordLayoutBuilder::ResizeLastBaseFieldIfNecessary(CharUnits offset) {
  // Check if we have a base to resize.
  if (!LastLaidOutBase.isValid())
    return false;

  // This offset does not overlap with the tail padding.
  if (offset >= NextFieldOffset)
    return false;

  // Restore the field offset and append an i8 array instead.
  FieldTypes.pop_back();
  NextFieldOffset = LastLaidOutBase.Offset;
  AppendBytes(LastLaidOutBase.NonVirtualSize);
  LastLaidOutBase.invalidate();

  return true;
}

llvm::Type *CGRecordLayoutBuilder::getByteArrayType(CharUnits numBytes) {
  assert(!numBytes.isZero() && "Empty byte arrays aren't allowed.");

  llvm::Type *Ty = llvm::Type::getInt8Ty(Types.getLLVMContext());
  if (numBytes > CharUnits::One())
    Ty = llvm::ArrayType::get(Ty, numBytes.getQuantity());

  return Ty;
}

void CGRecordLayoutBuilder::AppendBytes(CharUnits numBytes) {
  if (numBytes.isZero())
    return;

  // Append the padding field
  AppendField(NextFieldOffset, getByteArrayType(numBytes));
}

CharUnits CGRecordLayoutBuilder::getTypeAlignment(llvm::Type *Ty) const {
  if (Packed)
    return CharUnits::One();

  return CharUnits::fromQuantity(Types.getDataLayout().getABITypeAlignment(Ty));
}

CharUnits CGRecordLayoutBuilder::getAlignmentAsLLVMStruct() const {
  if (Packed)
    return CharUnits::One();

  CharUnits maxAlignment = CharUnits::One();
  for (size_t i = 0; i != FieldTypes.size(); ++i)
    maxAlignment = std::max(maxAlignment, getTypeAlignment(FieldTypes[i]));

  return maxAlignment;
}

/// Merge in whether a field of the given type is zero-initializable.
void CGRecordLayoutBuilder::CheckZeroInitializable(QualType T) {
  // This record already contains a member pointer.
  if (!IsZeroInitializableAsBase)
    return;

  // Can only have member pointers if we're compiling C++.
  if (!Types.getContext().getLangOpts().CPlusPlus)
    return;

  const Type *elementType = T->getBaseElementTypeUnsafe();

  if (const MemberPointerType *MPT = elementType->getAs<MemberPointerType>()) {
    if (!Types.getCXXABI().isZeroInitializable(MPT))
      IsZeroInitializable = IsZeroInitializableAsBase = false;
  } else if (const RecordType *RT = elementType->getAs<RecordType>()) {
    const CXXRecordDecl *RD = cast<CXXRecordDecl>(RT->getDecl());
    const CGRecordLayout &Layout = Types.getCGRecordLayout(RD);
    if (!Layout.isZeroInitializable())
      IsZeroInitializable = IsZeroInitializableAsBase = false;
  }
}

CGRecordLayout *CodeGenTypes::ComputeRecordLayout(const RecordDecl *D,
                                                  llvm::StructType *Ty) {
  CGRecordLayoutBuilder Builder(*this);

  Builder.Layout(D);

  Ty->setBody(Builder.FieldTypes, Builder.Packed);

  // If we're in C++, compute the base subobject type.
  llvm::StructType *BaseTy = 0;
  if (isa<CXXRecordDecl>(D) && !D->isUnion()) {
    BaseTy = Builder.BaseSubobjectType;
    if (!BaseTy) BaseTy = Ty;
  }

  CGRecordLayout *RL =
    new CGRecordLayout(Ty, BaseTy, Builder.IsZeroInitializable,
                       Builder.IsZeroInitializableAsBase);

  RL->NonVirtualBases.swap(Builder.NonVirtualBases);
  RL->CompleteObjectVirtualBases.swap(Builder.VirtualBases);

  // Add all the field numbers.
  RL->FieldInfo.swap(Builder.Fields);

  // Add bitfield info.
  RL->BitFields.swap(Builder.BitFields);

  // Dump the layout, if requested.
  if (getContext().getLangOpts().DumpRecordLayouts) {
    llvm::errs() << "\n*** Dumping IRgen Record Layout\n";
    llvm::errs() << "Record: ";
    D->dump();
    llvm::errs() << "\nLayout: ";
    RL->dump();
  }

#ifndef NDEBUG
  // Verify that the computed LLVM struct size matches the AST layout size.
  const ASTRecordLayout &Layout = getContext().getASTRecordLayout(D);

  uint64_t TypeSizeInBits = getContext().toBits(Layout.getSize());
  assert(TypeSizeInBits == getDataLayout().getTypeAllocSizeInBits(Ty) &&
         "Type size mismatch!");

  if (BaseTy) {
    CharUnits NonVirtualSize  = Layout.getNonVirtualSize();
    CharUnits NonVirtualAlign = Layout.getNonVirtualAlign();
    CharUnits AlignedNonVirtualTypeSize = 
      NonVirtualSize.RoundUpToAlignment(NonVirtualAlign);

    uint64_t AlignedNonVirtualTypeSizeInBits = 
      getContext().toBits(AlignedNonVirtualTypeSize);

    assert(AlignedNonVirtualTypeSizeInBits == 
           getDataLayout().getTypeAllocSizeInBits(BaseTy) &&
           "Type size mismatch!");
  }
                                     
  // Verify that the LLVM and AST field offsets agree.
  llvm::StructType *ST =
    dyn_cast<llvm::StructType>(RL->getLLVMType());
  const llvm::StructLayout *SL = getDataLayout().getStructLayout(ST);

  const ASTRecordLayout &AST_RL = getContext().getASTRecordLayout(D);
  RecordDecl::field_iterator it = D->field_begin();
  const FieldDecl *LastFD = 0;
  bool IsMsStruct = D->isMsStruct(getContext());
  for (unsigned i = 0, e = AST_RL.getFieldCount(); i != e; ++i, ++it) {
    const FieldDecl *FD = *it;

    // For non-bit-fields, just check that the LLVM struct offset matches the
    // AST offset.
    if (!FD->isBitField()) {
      unsigned FieldNo = RL->getLLVMFieldNo(FD);
      assert(AST_RL.getFieldOffset(i) == SL->getElementOffsetInBits(FieldNo) &&
             "Invalid field offset!");
      LastFD = FD;
      continue;
    }

    if (IsMsStruct) {
      // Zero-length bitfields following non-bitfield members are
      // ignored:
      if (getContext().ZeroBitfieldFollowsNonBitfield(FD, LastFD)) {
        --i;
        continue;
      }
      LastFD = FD;
    }
    
    // Ignore unnamed bit-fields.
    if (!FD->getDeclName()) {
      LastFD = FD;
      continue;
    }
    
    const CGBitFieldInfo &Info = RL->getBitFieldInfo(FD);
    for (unsigned i = 0, e = Info.getNumComponents(); i != e; ++i) {
      const CGBitFieldInfo::AccessInfo &AI = Info.getComponent(i);

      // Verify that every component access is within the structure.
      uint64_t FieldOffset = SL->getElementOffsetInBits(AI.FieldIndex);
      uint64_t AccessBitOffset = FieldOffset +
        getContext().toBits(AI.FieldByteOffset);
      assert(AccessBitOffset + AI.AccessWidth <= TypeSizeInBits &&
             "Invalid bit-field access (out of range)!");
    }
  }
#endif

  return RL;
}

void CGRecordLayout::print(raw_ostream &OS) const {
  OS << "<CGRecordLayout\n";
  OS << "  LLVMType:" << *CompleteObjectType << "\n";
  if (BaseSubobjectType)
    OS << "  NonVirtualBaseLLVMType:" << *BaseSubobjectType << "\n"; 
  OS << "  IsZeroInitializable:" << IsZeroInitializable << "\n";
  OS << "  BitFields:[\n";

  // Print bit-field infos in declaration order.
  std::vector<std::pair<unsigned, const CGBitFieldInfo*> > BFIs;
  for (llvm::DenseMap<const FieldDecl*, CGBitFieldInfo>::const_iterator
         it = BitFields.begin(), ie = BitFields.end();
       it != ie; ++it) {
    const RecordDecl *RD = it->first->getParent();
    unsigned Index = 0;
    for (RecordDecl::field_iterator
           it2 = RD->field_begin(); *it2 != it->first; ++it2)
      ++Index;
    BFIs.push_back(std::make_pair(Index, &it->second));
  }
  llvm::array_pod_sort(BFIs.begin(), BFIs.end());
  for (unsigned i = 0, e = BFIs.size(); i != e; ++i) {
    OS.indent(4);
    BFIs[i].second->print(OS);
    OS << "\n";
  }

  OS << "]>\n";
}

void CGRecordLayout::dump() const {
  print(llvm::errs());
}

void CGBitFieldInfo::print(raw_ostream &OS) const {
  OS << "<CGBitFieldInfo";
  OS << " Size:" << Size;
  OS << " IsSigned:" << IsSigned << "\n";

  OS.indent(4 + strlen("<CGBitFieldInfo"));
  OS << " NumComponents:" << getNumComponents();
  OS << " Components: [";
  if (getNumComponents()) {
    OS << "\n";
    for (unsigned i = 0, e = getNumComponents(); i != e; ++i) {
      const AccessInfo &AI = getComponent(i);
      OS.indent(8);
      OS << "<AccessInfo"
         << " FieldIndex:" << AI.FieldIndex
         << " FieldByteOffset:" << AI.FieldByteOffset.getQuantity()
         << " FieldBitStart:" << AI.FieldBitStart
         << " AccessWidth:" << AI.AccessWidth << "\n";
      OS.indent(8 + strlen("<AccessInfo"));
      OS << " AccessAlignment:" << AI.AccessAlignment.getQuantity()
         << " TargetBitOffset:" << AI.TargetBitOffset
         << " TargetBitWidth:" << AI.TargetBitWidth
         << ">\n";
    }
    OS.indent(4);
  }
  OS << "]>";
}

void CGBitFieldInfo::dump() const {
  print(llvm::errs());
}
