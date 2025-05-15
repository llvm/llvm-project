//==---- QualTypeMapper.cpp - Maps Clang QualType to LLVMABI Types ---------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Maps Clang QualType instances to corresponding LLVM ABI type
/// representations. This mapper translates high-level type information from the
/// AST into low-level ABI-specific types that encode size, alignment, and
/// layout details required for code generation and cross-language
/// interoperability.
///
//===----------------------------------------------------------------------===//
#include "clang/CodeGen/QualTypeMapper.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/ASTFwd.h"
#include "clang/AST/Attr.h"
#include "clang/AST/Decl.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/RecordLayout.h"
#include "clang/AST/Type.h"
#include "clang/Basic/AddressSpaces.h"
#include "clang/Basic/LLVM.h"
#include "clang/Basic/TargetInfo.h"
#include "llvm/ABI/Types.h"
#include "llvm/Support/Alignment.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/TypeSize.h"
#include "llvm/Support/raw_ostream.h"

namespace clang {
namespace CodeGen {

/// Main entry point for converting Clang QualType to LLVM ABI Type.
/// This method performs type canonicalization, caching, and dispatches
/// to specialized conversion methods based on the type kind.
///
/// \param QT The Clang QualType to convert
/// \return Corresponding LLVM ABI Type representation, or nullptr on error
const llvm::abi::Type *QualTypeMapper::convertType(QualType QT, bool InMemory) {
  // Canonicalize type and strip qualifiers
  // This ensures consistent type representation across different contexts
  QT = QT.getCanonicalType().getUnqualifiedType();

  // Results are cached since type conversion may be expensive
  auto It = TypeCache.find(QT);
  if (It != TypeCache.end() && !QT->isBooleanType())
    return It->second;

  const llvm::abi::Type *Result = nullptr;
  if (const auto *BT = dyn_cast<BuiltinType>(QT.getTypePtr()))
    Result = convertBuiltinType(BT, InMemory);
  else if (const auto *PT = dyn_cast<PointerType>(QT.getTypePtr()))
    Result = convertPointerType(PT);
  else if (const auto *RT = dyn_cast<ReferenceType>(QT.getTypePtr()))
    Result = convertReferenceType(RT);
  else if (const auto *AT = dyn_cast<ArrayType>(QT.getTypePtr()))
    Result = convertArrayType(AT);
  else if (const auto *VT = dyn_cast<VectorType>(QT.getTypePtr()))
    Result = convertVectorType(VT);
  else if (const auto *RT = dyn_cast<RecordType>(QT.getTypePtr()))
    Result = convertRecordType(RT);
  else if (const auto *ET = dyn_cast<EnumType>(QT.getTypePtr()))
    Result = convertEnumType(ET);
  else if (const auto *CT = dyn_cast<ComplexType>(QT.getTypePtr()))
    Result = convertComplexType(CT);
  else if (const auto *AT = dyn_cast<AtomicType>(QT.getTypePtr()))
    return convertType(AT->getValueType());
  else if (const auto *BPT = dyn_cast<BlockPointerType>(QT.getTypePtr()))
    return createPointerTypeForPointee(ASTCtx.VoidPtrTy);
  else if (const auto *PipeT = dyn_cast<PipeType>(QT.getTypePtr()))
    Result = createPointerTypeForPointee(ASTCtx.VoidPtrTy);
  else if (const auto *MT = dyn_cast<ConstantMatrixType>(QT.getTypePtr())) {
    const llvm::abi::Type *ElementType = convertType(MT->getElementType());
    uint64_t NumElements = MT->getNumRows() * MT->getNumColumns();
    return Builder.getArrayType(ElementType, NumElements, true);
  } else if (const auto *MPT = dyn_cast<MemberPointerType>(QT.getTypePtr())) {
    Result = convertMemberPointerType(MPT);
  } else if (const auto *BIT = dyn_cast<BitIntType>(QT.getTypePtr())) {
    unsigned RawNumBits = BIT->getNumBits();
    bool IsPromotableInt = BIT->getNumBits() < ASTCtx.getTypeSize(ASTCtx.IntTy);
    bool IsSigned = BIT->isSigned();
    llvm::Align TypeAlign = getTypeAlign(QT);
    return Builder.getIntegerType(RawNumBits, TypeAlign, IsSigned, false, true,
                                  IsPromotableInt, InMemory);
  } else if (isa<ObjCObjectType>(QT.getTypePtr()) ||
             isa<ObjCObjectPointerType>(QT.getTypePtr())) {
    // Objective-C objects are represented as pointers in the ABI
    auto PointerSize =
        ASTCtx.getTargetInfo().getPointerWidth(QT.getAddressSpace());
    llvm::Align PointerAlign =
        llvm::Align(ASTCtx.getTargetInfo().getPointerAlign(LangAS::Default));
    return Builder.getPointerType(
        PointerSize, llvm::Align(PointerAlign.value() / 8),
        ASTCtx.getTargetInfo().getTargetAddressSpace(QT.getAddressSpace()));
  } else
    QT.dump();

  TypeCache[QT] = Result;
  return Result;
}

/// Converts C/C++ builtin types to LLVM ABI types.
/// This handles all fundamental scalar types including integers, floats,
/// and special types like void and bool.
///
/// \param BT The BuiltinType to convert
/// \return Corresponding LLVM ABI integer, float, or void type
const llvm::abi::Type *QualTypeMapper::convertBuiltinType(const BuiltinType *BT,
                                                          bool InMemory) {
  QualType QT(BT, 0);

  switch (BT->getKind()) {
  case BuiltinType::Void:
    return Builder.getVoidType();

  case BuiltinType::NullPtr:
    return createPointerTypeForPointee(QT);

  case BuiltinType::Bool:
    return Builder.getIntegerType(1, getTypeAlign(QT), false, true, false,
                                  ASTCtx.isPromotableIntegerType(QT), InMemory);
  case BuiltinType::Char_S:
  case BuiltinType::Char_U:
  case BuiltinType::SChar:
  case BuiltinType::UChar:
  case BuiltinType::WChar_S:
  case BuiltinType::WChar_U:
  case BuiltinType::Char8:
  case BuiltinType::Char16:
  case BuiltinType::Char32:
  case BuiltinType::Short:
  case BuiltinType::UShort:
  case BuiltinType::Int:
  case BuiltinType::UInt:
  case BuiltinType::Long:
  case BuiltinType::ULong:
  case BuiltinType::LongLong:
  case BuiltinType::ULongLong:
  case BuiltinType::Int128:
  case BuiltinType::UInt128:
    return Builder.getIntegerType(ASTCtx.getTypeSize(QT), getTypeAlign(QT),
                                  BT->isSignedInteger(), false, false,
                                  ASTCtx.isPromotableIntegerType(QT));

  case BuiltinType::Half:
  case BuiltinType::Float16:
  case BuiltinType::BFloat16:
  case BuiltinType::Float:
  case BuiltinType::Double:
  case BuiltinType::LongDouble:
  case BuiltinType::Float128:
    return Builder.getFloatType(ASTCtx.getFloatTypeSemantics(QT),
                                getTypeAlign(QT));

  case BuiltinType::OCLImage1dRO:
  case BuiltinType::OCLImage1dWO:
  case BuiltinType::OCLImage1dRW:
  case BuiltinType::OCLImage1dArrayRO:
  case BuiltinType::OCLImage1dArrayWO:
  case BuiltinType::OCLImage1dArrayRW:
  case BuiltinType::OCLImage1dBufferRO:
  case BuiltinType::OCLImage1dBufferWO:
  case BuiltinType::OCLImage1dBufferRW:
  case BuiltinType::OCLImage2dRO:
  case BuiltinType::OCLImage2dWO:
  case BuiltinType::OCLImage2dRW:
  case BuiltinType::OCLImage2dArrayRO:
  case BuiltinType::OCLImage2dArrayWO:
  case BuiltinType::OCLImage2dArrayRW:
  case BuiltinType::OCLImage2dDepthRO:
  case BuiltinType::OCLImage2dDepthWO:
  case BuiltinType::OCLImage2dDepthRW:
  case BuiltinType::OCLImage2dArrayDepthRO:
  case BuiltinType::OCLImage2dArrayDepthWO:
  case BuiltinType::OCLImage2dArrayDepthRW:
  case BuiltinType::OCLImage2dMSAARO:
  case BuiltinType::OCLImage2dMSAAWO:
  case BuiltinType::OCLImage2dMSAARW:
  case BuiltinType::OCLImage2dArrayMSAARO:
  case BuiltinType::OCLImage2dArrayMSAAWO:
  case BuiltinType::OCLImage2dArrayMSAARW:
  case BuiltinType::OCLImage2dMSAADepthRO:
  case BuiltinType::OCLImage2dMSAADepthWO:
  case BuiltinType::OCLImage2dMSAADepthRW:
  case BuiltinType::OCLImage2dArrayMSAADepthRO:
  case BuiltinType::OCLImage2dArrayMSAADepthWO:
  case BuiltinType::OCLImage2dArrayMSAADepthRW:
  case BuiltinType::OCLImage3dRO:
  case BuiltinType::OCLImage3dWO:
  case BuiltinType::OCLImage3dRW:
    return createPointerTypeForPointee(QT);

  case BuiltinType::OCLSampler:
  case BuiltinType::OCLEvent:
  case BuiltinType::OCLQueue:
    return createPointerTypeForPointee(QT);

  default:
    // Unhandled BuiltinTypes are treated as unsigned integers.
    return Builder.getIntegerType(ASTCtx.getTypeSize(QualType(BT, 0)),
                                  getTypeAlign(QualType(BT, 0)), false);
  }
}

/// Converts array types to LLVM ABI array representations.
/// Handles different array kinds: constant arrays, incomplete arrays,
/// and variable-length arrays.
///
/// \param AT The ArrayType to convert
/// \return LLVM ABI ArrayType or PointerType
const llvm::abi::Type *
QualTypeMapper::convertArrayType(const clang::ArrayType *AT) {
  const llvm::abi::Type *ElementType = convertType(AT->getElementType());

  if (const auto *CAT = dyn_cast<ConstantArrayType>(AT)) {
    auto NumElements = CAT->getZExtSize();
    return Builder.getArrayType(ElementType, NumElements);
  }
  if (const auto *IAT = dyn_cast<IncompleteArrayType>(AT))
    return Builder.getArrayType(ElementType, 0);
  if (const auto *VAT = dyn_cast<VariableArrayType>(AT))
    return createPointerTypeForPointee(VAT->getPointeeType());
  // Fallback for other array types
  return Builder.getArrayType(ElementType, 1);
}

const llvm::abi::Type *QualTypeMapper::convertVectorType(const VectorType *VT) {
  const llvm::abi::Type *ElementType = convertType(VT->getElementType());
  QualType VectorQualType(VT, 0);

  // Handle element size adjustments for sub-byte types
  if (auto *IT = llvm::dyn_cast<llvm::abi::IntegerType>(ElementType)) {
    unsigned BW = IT->getSizeInBits().getFixedValue();
    if (BW != 1 && (BW & 7)) {
      BW = llvm::bit_ceil(BW);
      BW = std::clamp(BW, 8u, 64u);
      bool Signed = IT->isSigned();
      ElementType = Builder.getIntegerType(BW, llvm::Align(BW / 8), Signed);
    } else if (BW < 8 && BW != 1) {
      bool Signed = IT->isSigned();
      ElementType = Builder.getIntegerType(8, llvm::Align(1), Signed);
    }
  }

  unsigned NElems = VT->getNumElements();
  uint64_t LogicalSizeInBits =
      NElems * ElementType->getSizeInBits().getFixedValue();

  // Only round up for small vectors (≤ 64 bits)
  if (LogicalSizeInBits <= 64) {
    uint64_t ABISizeInBits = ASTCtx.getTypeSize(VectorQualType);
    if (ABISizeInBits > LogicalSizeInBits) {
      uint64_t ElementSizeInBits = ElementType->getSizeInBits().getFixedValue();
      NElems = ABISizeInBits / ElementSizeInBits;
    }
  }
  // For larger vectors, keep exact element count

  llvm::ElementCount NumElements = llvm::ElementCount::getFixed(NElems);
  llvm::Align VectorAlign = getTypeAlign(VectorQualType);

  return Builder.getVectorType(ElementType, NumElements, VectorAlign);
}

/// Converts complex types to LLVM ABI complex representations.
/// Complex types consist of two components of the element type
/// (real and imaginary parts).
///
/// \param CT The ComplexType to convert
/// \return LLVM ABI ComplexType with element type and alignment
const llvm::abi::Type *
QualTypeMapper::convertComplexType(const ComplexType *CT) {
  const llvm::abi::Type *ElementType = convertType(CT->getElementType());
  llvm::Align ComplexAlign = getTypeAlign(QualType(CT, 0));

  return Builder.getComplexType(ElementType, ComplexAlign);
}

/// Converts member pointer types to LLVM ABI representations.
/// Member pointers have different layouts depending on whether they
/// point to functions or data members.
///
/// \param MPT The MemberPointerType to convert
/// \return LLVM ABI MemberPointerType
const llvm::abi::Type *
QualTypeMapper::convertMemberPointerType(const clang::MemberPointerType *MPT) {
  QualType QT(MPT, 0);
  uint64_t Size = ASTCtx.getTypeSize(QT);
  llvm::Align Align = getTypeAlign(QT);

  bool IsFunctionPointer = MPT->isMemberFunctionPointerType();

  return Builder.getMemberPointerType(IsFunctionPointer, Size, Align);
}

/// Converts record types (struct/class/union) to LLVM ABI representations.
/// This is the main dispatch method that handles different record kinds
/// and delegates to specialized converters.
///
/// \param RT The RecordType to convert
/// \return LLVM ABI StructType or UnionType
const llvm::abi::Type *QualTypeMapper::convertRecordType(const RecordType *RT) {
  const RecordDecl *RD = RT->getOriginalDecl()->getDefinition();
  bool canPassInRegs = false;
  bool hasFlexibleArrMember = false;
  if (RD) {
    canPassInRegs = RD->canPassInRegisters();
    hasFlexibleArrMember = RD->hasFlexibleArrayMember();
  }
  if (!RD) {
    SmallVector<llvm::abi::FieldInfo, 0> Fields;
    return Builder.getStructType(
        Fields, llvm::TypeSize::getFixed(0), llvm::Align(1),
        llvm::abi::StructPacking::Default, {}, {}, false, false, false, false,
        hasFlexibleArrMember, false, canPassInRegs);
  }

  if (RD->isUnion()) {
    const RecordDecl *UD = RT->getOriginalDecl();
    if (UD->hasAttr<TransparentUnionAttr>())
      return convertUnionType(RD, true);
    return convertUnionType(RD);
  }

  // Handle C++ classes with base classes
  auto *const CXXRd = dyn_cast<CXXRecordDecl>(RD);
  if (CXXRd && (CXXRd->getNumBases() > 0 || CXXRd->getNumVBases() > 0)) {
    return convertCXXRecordType(CXXRd, canPassInRegs);
  }
  return convertStructType(RD);
}

/// Converts C++ classes with inheritance to LLVM ABI struct representations.
/// This method handles the complex layout of C++ objects including:
/// - Virtual table pointers for polymorphic classes
/// - Base class subobjects (both direct and virtual bases)
/// - Member field layout with proper offsets
///
/// \param RD The C++ record declaration
/// \return LLVM ABI StructType representing the complete object layout
const llvm::abi::StructType *
QualTypeMapper::convertCXXRecordType(const CXXRecordDecl *RD,
                                     bool canPassInRegs) {
  const ASTRecordLayout &Layout = ASTCtx.getASTRecordLayout(RD);
  SmallVector<llvm::abi::FieldInfo, 16> Fields;
  SmallVector<llvm::abi::FieldInfo, 8> BaseClasses;
  SmallVector<llvm::abi::FieldInfo, 8> VirtualBaseClasses;

  // Add vtable pointer for polymorphic classes
  if (RD->isPolymorphic()) {
    const llvm::abi::Type *VtablePointer =
        createPointerTypeForPointee(ASTCtx.VoidPtrTy);
    Fields.emplace_back(VtablePointer, 0);
  }

  for (const auto &Base : RD->bases()) {
    if (Base.isVirtual())
      continue;

    const RecordType *BaseRT = Base.getType()->castAs<RecordType>();
    const llvm::abi::Type *BaseType = convertType(Base.getType());
    uint64_t BaseOffset =
        Layout.getBaseClassOffset(BaseRT->getAsCXXRecordDecl()).getQuantity() *
        8;

    BaseClasses.emplace_back(BaseType, BaseOffset);
  }

  for (const auto &VBase : RD->vbases()) {
    const RecordType *VBaseRT = VBase.getType()->getAs<RecordType>();
    if (!VBaseRT)
      continue;

    const llvm::abi::Type *VBaseType = convertType(VBase.getType());
    uint64_t VBaseOffset =
        Layout.getVBaseClassOffset(VBaseRT->getAsCXXRecordDecl())
            .getQuantity() *
        8;

    VirtualBaseClasses.emplace_back(VBaseType, VBaseOffset);
  }

  computeFieldInfo(RD, Fields, Layout);

  llvm::sort(Fields,
             [](const llvm::abi::FieldInfo &A, const llvm::abi::FieldInfo &B) {
               return A.OffsetInBits < B.OffsetInBits;
             });

  llvm::TypeSize Size =
      llvm::TypeSize::getFixed(Layout.getSize().getQuantity() * 8);
  llvm::Align Alignment = llvm::Align(Layout.getAlignment().getQuantity());

  bool HasNonTrivialCopy = !RD->hasSimpleCopyConstructor();
  bool HasNonTrivialDtor = !RD->hasSimpleDestructor();
  bool HasFlexibleArrayMember = RD->hasFlexibleArrayMember();
  bool HasUnalignedFields = false;

  unsigned FieldIndex = 0;
  for (const auto *FD : RD->fields()) {
    uint64_t FieldOffset = Layout.getFieldOffset(FieldIndex);
    uint64_t ExpectedAlignment = ASTCtx.getTypeAlign(FD->getType());
    if (FieldOffset % ExpectedAlignment != 0) {
      HasUnalignedFields = true;
      break;
    }
    ++FieldIndex;
  }

  return Builder.getStructType(
      Fields, Size, Alignment, llvm::abi::StructPacking::Default, BaseClasses,
      VirtualBaseClasses, true, RD->isPolymorphic(), HasNonTrivialCopy,
      HasNonTrivialDtor, HasFlexibleArrayMember, HasUnalignedFields,
      canPassInRegs);
}

/// Converts reference types to pointer representations in the ABI.
/// Both lvalue references (T&) and rvalue references (T&&) are represented
/// as pointers at the ABI level.
///
/// \param RT The ReferenceType to convert
/// \return LLVM ABI PointerType
const llvm::abi::Type *
QualTypeMapper::convertReferenceType(const ReferenceType *RT) {
  return createPointerTypeForPointee(RT->getPointeeType());
}

/// Converts pointer types to LLVM ABI pointer representations.
/// Takes into account address space information for the pointed-to type.
///
/// \param PT The PointerType to convert
/// \return LLVM ABI PointerType with appropriate size and alignment
const llvm::abi::Type *
QualTypeMapper::convertPointerType(const clang::PointerType *PT) {
  return createPointerTypeForPointee(PT->getPointeeType());
}

/// Converts enumeration types to their underlying integer representations.
/// This method handles various enum states and falls back to safe defaults
/// when enum information is incomplete or invalid.
///
/// \param ET The EnumType to convert
/// \return LLVM ABI IntegerType representing the enum's underlying type
const llvm::abi::Type *
QualTypeMapper::convertEnumType(const clang::EnumType *ET) {
  const EnumDecl *ED = ET->getOriginalDecl();
  QualType UnderlyingType = ED->getIntegerType();

  if (UnderlyingType.isNull())
    UnderlyingType = ASTCtx.IntTy;

  return convertType(UnderlyingType);
}

/// Converts plain C structs and C++ classes without inheritance.
/// This handles the simpler case where we only need to layout member fields
/// without considering base classes or virtual functions.
///
/// \param RD The RecordDecl to convert
/// \return LLVM ABI StructType
const llvm::abi::StructType *
QualTypeMapper::convertStructType(const clang::RecordDecl *RD) {
  const ASTRecordLayout &Layout = ASTCtx.getASTRecordLayout(RD);

  bool IsCXXRecord = isa<CXXRecordDecl>(RD);
  SmallVector<llvm::abi::FieldInfo, 16> Fields;
  computeFieldInfo(RD, Fields, Layout);

  llvm::TypeSize Size =
      llvm::TypeSize::getFixed(Layout.getSize().getQuantity() * 8);
  llvm::Align Alignment = llvm::Align(Layout.getAlignment().getQuantity());

  return Builder.getStructType(
      Fields, Size, Alignment, llvm::abi::StructPacking::Default, {}, {},
      IsCXXRecord, false, false, false, RD->hasFlexibleArrayMember(), false,
      RD->canPassInRegisters());
}

/// Converts C union types where all fields occupy the same memory location.
/// The union size is determined by its largest member, and all fields
/// start at offset 0.
///
/// \param RD The RecordDecl representing the union
/// \return LLVM ABI UnionType
const llvm::abi::StructType *
QualTypeMapper::convertUnionType(const clang::RecordDecl *RD,
                                 bool isTransparent) {
  const ASTRecordLayout &Layout = ASTCtx.getASTRecordLayout(RD);

  SmallVector<llvm::abi::FieldInfo, 16> AllFields;
  computeFieldInfo(RD, AllFields, Layout);

  llvm::TypeSize Size =
      llvm::TypeSize::getFixed(Layout.getSize().getQuantity() * 8);
  llvm::Align Alignment = llvm::Align(Layout.getAlignment().getQuantity());

auto *UnionMeta = Builder.createUnionMetadata(AllFields, Size, Alignment);
  const llvm::abi::Type *StorageType = nullptr;
  bool SeenNamedMember = false;

  for (const auto *Field : RD->fields()) {
    if (Field->isBitField() && Field->isZeroLengthBitField())
      continue;

    const llvm::abi::Type *FieldType = convertType(Field->getType(), true);

    if (!SeenNamedMember) {
      SeenNamedMember = Field->getIdentifier() != nullptr;
      if (!SeenNamedMember) {
        if (const auto *FieldRD = Field->getType()->getAsRecordDecl())
          SeenNamedMember = FieldRD->findFirstNamedDataMember() != nullptr;
      }

      if (SeenNamedMember) {
        StorageType = FieldType;
      }
    }

    if (!StorageType ||
        FieldType->getAlignment().value() >
            StorageType->getAlignment().value() ||
        (FieldType->getAlignment().value() ==
             StorageType->getAlignment().value() &&
         FieldType->getSizeInBits().getFixedValue() >
             StorageType->getSizeInBits().getFixedValue())) {
      StorageType = FieldType;
    }
  }

  SmallVector<llvm::abi::FieldInfo, 1> UnionFields;
  if (StorageType) {
    UnionFields.emplace_back(StorageType, 0, false, 0, false);
  }

  const llvm::abi::StructType *LoweredUnion = Builder.getUnionType(
      UnionFields, Size, Alignment, llvm::abi::StructPacking::Default,
      isTransparent, RD->canPassInRegisters(),isa<CXXRecordDecl>(RD));

  LoweredUnion->setMetadata(UnionMeta);

  return LoweredUnion;
}

llvm::Align QualTypeMapper::getPreferredTypeAlign(QualType QT) const {
  return llvm::Align(ASTCtx.getPreferredTypeAlignInChars(QT).getQuantity());
}

llvm::Align QualTypeMapper::getTypeAlign(QualType QT) const {

  return llvm::Align(ASTCtx.getTypeAlignInChars(QT).getQuantity());
}

const llvm::abi::Type *
QualTypeMapper::createPointerTypeForPointee(QualType PointeeType) {
  auto AddrSpace = PointeeType.getAddressSpace();
  auto PointerSize = ASTCtx.getTargetInfo().getPointerWidth(AddrSpace);
  llvm::Align Alignment =
      llvm::Align(ASTCtx.getTargetInfo().getPointerAlign(AddrSpace));
  return Builder.getPointerType(
      PointerSize, llvm::Align(Alignment.value() / 8),
      ASTCtx.getTargetInfo().getTargetAddressSpace(AddrSpace));
}

/// Processes the fields of a record (struct/class/union) and populates
/// the Fields vector with FieldInfo objects containing type, offset,
/// and bitfield information.
///
/// \param RD The RecordDecl whose fields to process
/// \param Fields Output vector to populate with field information
/// \param Layout The AST record layout containing field offset information
void QualTypeMapper::computeFieldInfo(
    const RecordDecl *RD, SmallVectorImpl<llvm::abi::FieldInfo> &Fields,
    const ASTRecordLayout &Layout) {
  unsigned FieldIndex = 0;

  for (const auto *FD : RD->fields()) {
    const llvm::abi::Type *FieldType = convertType(FD->getType(), true);
    uint64_t OffsetInBits = Layout.getFieldOffset(FieldIndex);

    bool IsBitField = FD->isBitField();
    uint64_t BitFieldWidth = 0;
    bool IsUnnamed = false;

    if (IsBitField) {
      BitFieldWidth = FD->getBitWidthValue();
      IsUnnamed = FD->isUnnamedBitField();
    }

    Fields.emplace_back(FieldType, OffsetInBits, IsBitField, BitFieldWidth,
                        IsUnnamed);
    ++FieldIndex;
  }
}

} // namespace CodeGen
} // namespace clang
