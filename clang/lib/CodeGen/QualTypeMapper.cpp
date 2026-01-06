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
#include "QualTypeMapper.h"
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
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/TypeSize.h"
#include <cstdint>

namespace clang {
namespace CodeGen {

/// Main entry point for converting Clang QualType to LLVM ABI Type.
/// This method performs type canonicalization, caching, and dispatches
/// to specialized conversion methods based on the type kind.
///
/// \param QT The Clang QualType to convert
/// \return Corresponding LLVM ABI Type representation, or nullptr on error
const llvm::abi::Type *QualTypeMapper::convertType(QualType QT) {
  // Canonicalize type and strip qualifiers
  // This ensures consistent type representation across different contexts
  QT = QT.getCanonicalType().getUnqualifiedType();

  // Results are cached since type conversion may be expensive
  auto It = TypeCache.find(QT);
  if (It != TypeCache.end())
    return It->second;

  const llvm::abi::Type *Result = nullptr;
  switch (QT->getTypeClass()) {
    // Non-canonical and dependent types should have been stripped by
    // getCanonicalType() above or cannot appear during code generation.
#define TYPE(Class, Base)
#define ABSTRACT_TYPE(Class, Base)
#define NON_CANONICAL_TYPE(Class, Base) case Type::Class:
#define DEPENDENT_TYPE(Class, Base) case Type::Class:
#define NON_CANONICAL_UNLESS_DEPENDENT_TYPE(Class, Base) case Type::Class:
#include "clang/AST/TypeNodes.inc"
    llvm::reportFatalInternalError(
        "Non-canonical or dependent types should not reach ABI lowering");

  case Type::Builtin:
    Result = convertBuiltinType(cast<BuiltinType>(QT));
    break;
  case Type::Pointer:
    Result =
        createPointerTypeForPointee(cast<PointerType>(QT)->getPointeeType());
    break;
  case Type::LValueReference:
  case Type::RValueReference:
    Result =
        createPointerTypeForPointee(cast<ReferenceType>(QT)->getPointeeType());
    break;
  case Type::ConstantArray:
  case Type::ArrayParameter:
  case Type::IncompleteArray:
  case Type::VariableArray:
    Result = convertArrayType(cast<ArrayType>(QT));
    break;
  case Type::Vector:
  case Type::ExtVector:
    Result = convertVectorType(cast<VectorType>(QT));
    break;
  case Type::Record:
    Result = convertRecordType(cast<RecordType>(QT));
    break;
  case Type::Enum:
    Result = convertEnumType(cast<EnumType>(QT));
    break;
  case Type::Complex:
    Result = convertComplexType(cast<ComplexType>(QT));
    break;
  case Type::Atomic:
    return convertType(cast<AtomicType>(QT)->getValueType());
  case Type::BlockPointer:
    return createPointerTypeForPointee(ASTCtx.VoidPtrTy);
  case Type::Pipe:
    Result = createPointerTypeForPointee(ASTCtx.VoidPtrTy);
    break;
  case Type::ConstantMatrix: {
    const auto *MT = cast<ConstantMatrixType>(QT);
    return Builder.getArrayType(convertType(MT->getElementType()),
                                MT->getNumRows() * MT->getNumColumns(),
                                ASTCtx.getTypeSize(QT), /*IsMatrixType=*/true);
  }
  case Type::MemberPointer:
    Result = convertMemberPointerType(cast<MemberPointerType>(QT));
    break;
  case Type::BitInt: {
    const auto *BIT = cast<BitIntType>(QT);
    return Builder.getIntegerType(BIT->getNumBits(), getTypeAlign(QT),
                                  /*Signed=*/BIT->isSigned(),
                                  /*IsBitInt=*/true);
  }
  case Type::ObjCObject:
  case Type::ObjCInterface:
  case Type::ObjCObjectPointer:
    // Objective-C objects are represented as pointers in the ABI.
    return Builder.getPointerType(
        ASTCtx.getTargetInfo().getPointerWidth(QT.getAddressSpace()),
        llvm::Align(ASTCtx.getTargetInfo().getPointerAlign(LangAS::Default) /
                    8),
        ASTCtx.getTargetInfo().getTargetAddressSpace(QT.getAddressSpace()));
  case Type::Auto:
  case Type::DeducedTemplateSpecialization:
  case Type::FunctionProto:
  case Type::FunctionNoProto:
  case Type::HLSLAttributedResource:
    llvm::reportFatalInternalError("Type not supported in ABI lowering");
  }

  assert(Result && "convertType returned nullptr");
  TypeCache[QT] = Result;
  return Result;
}

/// Converts C/C++ builtin types to LLVM ABI types.
/// This handles all fundamental scalar types including integers, floats,
/// and special types like void and bool.
///
/// \param BT The BuiltinType to convert
/// \return Corresponding LLVM ABI integer, float, or void type
const llvm::abi::Type *
QualTypeMapper::convertBuiltinType(const BuiltinType *BT) {
  QualType QT(BT, 0);

  switch (BT->getKind()) {
  case BuiltinType::Void:
    return Builder.getVoidType();

  case BuiltinType::NullPtr:
    return createPointerTypeForPointee(QT);

  case BuiltinType::Bool:
    return Builder.getIntegerType(1, getTypeAlign(QT), /*Signed=*/false,
                                  /*IsBitInt=*/false);

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
                                  /*Signed=*/BT->isSignedInteger(),
                                  /*IsBitInt=*/false);

  case BuiltinType::Half:
  case BuiltinType::Float16:
  case BuiltinType::BFloat16:
  case BuiltinType::Float:
  case BuiltinType::Double:
  case BuiltinType::LongDouble:
  case BuiltinType::Float128:
    return Builder.getFloatType(ASTCtx.getFloatTypeSemantics(QT),
                                getTypeAlign(QT));

  // TODO: IBM 128-bit extended double
  case BuiltinType::Ibm128:
    llvm::reportFatalInternalError(
        "IBM128 is not yet supported in the ABI lowering libary");

  // TODO: Fixed-point types
  case BuiltinType::ShortAccum:
  case BuiltinType::Accum:
  case BuiltinType::LongAccum:
  case BuiltinType::UShortAccum:
  case BuiltinType::UAccum:
  case BuiltinType::ULongAccum:
  case BuiltinType::ShortFract:
  case BuiltinType::Fract:
  case BuiltinType::LongFract:
  case BuiltinType::UShortFract:
  case BuiltinType::UFract:
  case BuiltinType::ULongFract:
  case BuiltinType::SatShortAccum:
  case BuiltinType::SatAccum:
  case BuiltinType::SatLongAccum:
  case BuiltinType::SatUShortAccum:
  case BuiltinType::SatUAccum:
  case BuiltinType::SatULongAccum:
  case BuiltinType::SatShortFract:
  case BuiltinType::SatFract:
  case BuiltinType::SatLongFract:
  case BuiltinType::SatUShortFract:
  case BuiltinType::SatUFract:
  case BuiltinType::SatULongFract:
    llvm::reportFatalInternalError(
        "Fixed Point types not yet implemented in the ABI lowering library");

    // OpenCL image types are represented as opaque pointers.
#define IMAGE_TYPE(ImgType, Id, SingletonId, Access, Suffix)                   \
  case BuiltinType::Id:
#include "clang/Basic/OpenCLImageTypes.def"
    // OpenCL extension types are represented as opaque pointers.
#define EXT_OPAQUE_TYPE(ExtType, Id, Ext) case BuiltinType::Id:
#include "clang/Basic/OpenCLExtensionTypes.def"
  case BuiltinType::OCLSampler:
  case BuiltinType::OCLEvent:
  case BuiltinType::OCLClkEvent:
  case BuiltinType::OCLQueue:
  case BuiltinType::OCLReserveID:
    return createPointerTypeForPointee(QT);

  // Objective-C builtin types are represented as opaque pointers.
  case BuiltinType::ObjCId:
  case BuiltinType::ObjCClass:
  case BuiltinType::ObjCSel:
    return createPointerTypeForPointee(QT);

    // Target-specific vector/matrix types — not yet implemented.
#define SVE_TYPE(Name, Id, SingletonId) case BuiltinType::Id:
#include "clang/Basic/AArch64ACLETypes.def"
    llvm::reportFatalInternalError(
        "AArch64 SVE types not yet supported in ABI lowering library");
#define PPC_VECTOR_TYPE(Name, Id, Size) case BuiltinType::Id:
#include "clang/Basic/PPCTypes.def"
    llvm::reportFatalInternalError(
        "PPC MMA types not yet supported in ABI lowering library");
#define RVV_TYPE(Name, Id, SingletonId) case BuiltinType::Id:
#include "clang/Basic/RISCVVTypes.def"
    llvm::reportFatalInternalError(
        "RISC-V vector types not yet supported in ABI lowering library");
#define WASM_TYPE(Name, Id, SingletonId) case BuiltinType::Id:
#include "clang/Basic/WebAssemblyReferenceTypes.def"
    llvm::reportFatalInternalError("WebAssembly reference types not yet "
                                   "supported in ABI lowering library");
#define AMDGPU_TYPE(Name, Id, SingletonId, Width, Align) case BuiltinType::Id:
#include "clang/Basic/AMDGPUTypes.def"
    llvm::reportFatalInternalError(
        "AMDGPU types not yet supported in ABI lowering library");
#define HLSL_INTANGIBLE_TYPE(Name, Id, SingletonId) case BuiltinType::Id:
#include "clang/Basic/HLSLIntangibleTypes.def"
    llvm::reportFatalInternalError(
        "HLSL intangible types not yet Supported in ABI lowering library");

    // Placeholder types should never reach ABI lowering.
#define PLACEHOLDER_TYPE(Id, SingletonId) case BuiltinType::Id:
#define BUILTIN_TYPE(Id, SingletonId)
#include "clang/AST/BuiltinTypes.def"
    llvm::reportFatalInternalError(
        "Placeholder type should not reach ABI lowering");
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
  uint64_t Size = ASTCtx.getTypeSize(AT);

  if (const auto *CAT = dyn_cast<ConstantArrayType>(AT)) {
    auto NumElements = CAT->getZExtSize();
    return Builder.getArrayType(ElementType, NumElements, Size);
  }
  if (isa<IncompleteArrayType>(AT))
    return Builder.getArrayType(ElementType, 0, 0);
  if (const auto *VAT = dyn_cast<VariableArrayType>(AT))
    return createPointerTypeForPointee(VAT->getPointeeType());
  llvm::reportFatalInternalError(
      "unexpected array type in ABI lowering (dependent array types should be "
      "resolved before reaching this point)");
}

const llvm::abi::Type *QualTypeMapper::convertVectorType(const VectorType *VT) {
  const llvm::abi::Type *ElementType = convertType(VT->getElementType());
  QualType VectorQualType(VT, 0);

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
/// \return LLVM ABI RecordType
const llvm::abi::Type *QualTypeMapper::convertRecordType(const RecordType *RT) {
  const RecordDecl *RD = RT->getDecl()->getDefinition();
  if (!RD)
    return Builder.getRecordType({}, llvm::TypeSize::getFixed(0),
                                 llvm::Align(1));

  if (RD->isUnion())
    return convertUnionType(RD, RD->hasAttr<TransparentUnionAttr>());

  // Handle C++ classes with base classes
  auto *CXXRd = dyn_cast<CXXRecordDecl>(RD);
  if (CXXRd && (CXXRd->getNumBases() > 0 || CXXRd->getNumVBases() > 0))
    return convertCXXRecordType(CXXRd);
  return convertStructType(RD);
}

/// Converts C++ classes with inheritance to LLVM ABI struct representations.
/// This method handles the complex layout of C++ objects including:
/// - Virtual table pointers for polymorphic classes
/// - Base class subobjects (both direct and virtual bases)
/// - Member field layout with proper offsets
///
/// \param RD The C++ record declaration
/// \return LLVM ABI RecordType representing the complete object layout
const llvm::abi::RecordType *
QualTypeMapper::convertCXXRecordType(const CXXRecordDecl *RD) {
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
    const RecordType *VBaseRT = VBase.getType()->castAs<RecordType>();
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

  llvm::abi::RecordFlags RecFlags = llvm::abi::RecordFlags::IsCXXRecord;
  if (RD->isPolymorphic())
    RecFlags |= llvm::abi::RecordFlags::IsPolymorphic;
  if (RD->canPassInRegisters())
    RecFlags |= llvm::abi::RecordFlags::CanPassInRegisters;
  if (RD->hasFlexibleArrayMember())
    RecFlags |= llvm::abi::RecordFlags::HasFlexibleArrayMember;

  return Builder.getRecordType(Fields, Size, Alignment,
                               llvm::abi::StructPacking::Default, BaseClasses,
                               VirtualBaseClasses, RecFlags);
}

/// Converts enumeration types to their underlying integer representations.
/// This method handles various enum states and falls back to safe defaults
/// when enum information is incomplete or invalid.
///
/// \param ET The EnumType to convert
/// \return LLVM ABI IntegerType representing the enum's underlying type
const llvm::abi::Type *
QualTypeMapper::convertEnumType(const clang::EnumType *ET) {
  const EnumDecl *ED = ET->getDecl();
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
/// \return LLVM ABI RecordType
const llvm::abi::RecordType *
QualTypeMapper::convertStructType(const clang::RecordDecl *RD) {
  const ASTRecordLayout &Layout = ASTCtx.getASTRecordLayout(RD);

  bool IsCXXRecord = isa<CXXRecordDecl>(RD);
  SmallVector<llvm::abi::FieldInfo, 16> Fields;
  computeFieldInfo(RD, Fields, Layout);

  llvm::TypeSize Size =
      llvm::TypeSize::getFixed(Layout.getSize().getQuantity() * 8);
  llvm::Align Alignment = llvm::Align(Layout.getAlignment().getQuantity());

  llvm::abi::RecordFlags RecFlags = llvm::abi::RecordFlags::None;
  if (IsCXXRecord)
    RecFlags |= llvm::abi::RecordFlags::IsCXXRecord;
  if (RD->canPassInRegisters())
    RecFlags |= llvm::abi::RecordFlags::CanPassInRegisters;
  if (RD->hasFlexibleArrayMember())
    RecFlags |= llvm::abi::RecordFlags::HasFlexibleArrayMember;

  return Builder.getRecordType(Fields, Size, Alignment,
                               llvm::abi::StructPacking::Default, {}, {},
                               RecFlags);
}

/// Converts C union types where all fields occupy the same memory location.
/// The union size is determined by its largest member, and all fields
/// start at offset 0.
///
/// \param RD The RecordDecl representing the union
/// \return LLVM ABI UnionType
const llvm::abi::RecordType *
QualTypeMapper::convertUnionType(const clang::RecordDecl *RD,
                                 bool isTransparent) {
  const ASTRecordLayout &Layout = ASTCtx.getASTRecordLayout(RD);

  SmallVector<llvm::abi::FieldInfo, 16> AllFields;
  computeFieldInfo(RD, AllFields, Layout);

  llvm::TypeSize Size =
      llvm::TypeSize::getFixed(Layout.getSize().getQuantity() * 8);
  llvm::Align Alignment = llvm::Align(Layout.getAlignment().getQuantity());

  llvm::abi::RecordFlags RecFlags = llvm::abi::RecordFlags::None;
  if (isTransparent)
    RecFlags |= llvm::abi::RecordFlags::IsTransparent;
  if (RD->canPassInRegisters())
    RecFlags |= llvm::abi::RecordFlags::CanPassInRegisters;
  if (isa<CXXRecordDecl>(RD))
    RecFlags |= llvm::abi::RecordFlags::IsCXXRecord;

  return Builder.getUnionType(AllFields, Size, Alignment,
                              llvm::abi::StructPacking::Default, RecFlags);
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
  // Function types without an explicit address space qualifier use the program
  // address space, which may differ from the default data address space on
  // targets like AMDGPU.
  unsigned TargetAddrSpace =
      PointeeType->isFunctionType() && !PointeeType.hasAddressSpace()
          ? DL.getProgramAddressSpace()
          : ASTCtx.getTargetInfo().getTargetAddressSpace(AddrSpace);
  return Builder.getPointerType(PointerSize, llvm::Align(Alignment.value() / 8),
                                TargetAddrSpace);
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
    const llvm::abi::Type *FieldType = convertType(FD->getType());
    uint64_t OffsetInBits = Layout.getFieldOffset(FieldIndex);

    bool IsBitField = FD->isBitField();
    uint64_t BitFieldWidth = 0;
    bool IsUnnamedBitField = false;

    if (IsBitField) {
      BitFieldWidth = FD->getBitWidthValue();
      IsUnnamedBitField = FD->isUnnamedBitField();
    }

    Fields.emplace_back(FieldType, OffsetInBits, IsBitField, BitFieldWidth,
                        IsUnnamedBitField);
    ++FieldIndex;
  }
}

} // namespace CodeGen
} // namespace clang
