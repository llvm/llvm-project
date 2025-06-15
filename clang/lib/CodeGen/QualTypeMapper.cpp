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
#include "clang/AST/Decl.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/RecordLayout.h"
#include "clang/AST/Type.h"
#include "clang/Basic/AddressSpaces.h"
#include "clang/Basic/LLVM.h"
#include "clang/Basic/TargetInfo.h"
#include "llvm/ABI/Types.h"
#include "llvm/Support/Alignment.h"
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
const llvm::abi::Type *QualTypeMapper::convertType(QualType QT) {
  // Canonicalize type and strip qualifiers
  // This ensures consistent type representation across different contexts
  QT = QT.getCanonicalType().getUnqualifiedType();

  // Results are cached since type conversion may be expensive
  auto It = TypeCache.find(QT);
  if (It != TypeCache.end())
    return It->second;

  const llvm::abi::Type *Result = nullptr;
  if (const auto *BT = dyn_cast<BuiltinType>(QT.getTypePtr())) {
    Result = convertBuiltinType(BT);
  } else if (const auto *PT = dyn_cast<PointerType>(QT.getTypePtr())) {
    Result = convertPointerType(PT);
  } else if (const auto *RT = dyn_cast<ReferenceType>(QT.getTypePtr())) {
    Result = convertReferenceType(RT);
  } else if (const auto *AT = dyn_cast<ArrayType>(QT.getTypePtr())) {
    Result = convertArrayType(AT);
  } else if (const auto *VT = dyn_cast<VectorType>(QT.getTypePtr())) {
    Result = convertVectorType(VT);
  } else if (const auto *RT = dyn_cast<RecordType>(QT.getTypePtr())) {
    Result = convertRecordType(RT);
  } else if (const auto *ET = dyn_cast<EnumType>(QT.getTypePtr())) {
    Result = convertEnumType(ET);
  } else if (const auto *BIT = dyn_cast<BitIntType>(QT.getTypePtr())) {
    // Handle C23 _BitInt(N) types - arbitrary precision integers
    QualType QT(BIT, 0);
    uint64_t NumBits = BIT->getNumBits();
    bool IsSigned = BIT->isSigned();
    llvm::Align TypeAlign = getTypeAlign(QT);
    return Builder.getIntegerType(NumBits, TypeAlign, IsSigned);
  } else if (isa<ObjCObjectType>(QT.getTypePtr()) ||
             isa<ObjCObjectPointerType>(QT.getTypePtr())) {
    // Objective-C objects are represented as pointers in the ABI
    auto PointerSize = ASTCtx.getTargetInfo().getPointerWidth(LangAS::Default);
    llvm::Align PointerAlign =
        llvm::Align(ASTCtx.getTargetInfo().getPointerAlign(LangAS::Default));
    return Builder.getPointerType(PointerSize, PointerAlign);
  } else {
    llvm_unreachable("Unsupported type for ABI lowering");
  }
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

  case BuiltinType::Bool:
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
                                  BT->isSignedInteger());

  case BuiltinType::Half:
  case BuiltinType::Float16:
  case BuiltinType::BFloat16:
  case BuiltinType::Float:
  case BuiltinType::Double:
  case BuiltinType::LongDouble:
  case BuiltinType::Float128:
    return Builder.getFloatType(ASTCtx.getFloatTypeSemantics(QT),
                                getTypeAlign(QT));

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

/// Converts vector types to LLVM ABI vector representations.
///
/// \param VT The VectorType to convert
/// \return LLVM ABI VectorType with element type, count, and alignment
const llvm::abi::Type *QualTypeMapper::convertVectorType(const VectorType *VT) {
  const llvm::abi::Type *ElementType = convertType(VT->getElementType());
  llvm::ElementCount NumElements =
      llvm::ElementCount::getFixed(VT->getNumElements());

  llvm::Align VectorAlign = getTypeAlign(QualType(VT, 0));

  return Builder.getVectorType(ElementType, NumElements, VectorAlign);
}

/// Converts record types (struct/class/union) to LLVM ABI representations.
/// This is the main dispatch method that handles different record kinds
/// and delegates to specialized converters.
///
/// \param RT The RecordType to convert
/// \return LLVM ABI StructType or UnionType
const llvm::abi::Type *QualTypeMapper::convertRecordType(const RecordType *RT) {
  const RecordDecl *RD = RT->getDecl()->getDefinition();
  if (!RD) {
    SmallVector<llvm::abi::FieldInfo, 0> Fields;
    return Builder.getStructType(Fields, llvm::TypeSize::getFixed(0),
                                 llvm::Align(1));
  }

  if (RD->isUnion())
    return convertUnionType(RD);

  // Handle C++ classes with base classes
  auto *const CXXRd = dyn_cast<CXXRecordDecl>(RD);
  if (CXXRd && (CXXRd->getNumBases() > 0 || CXXRd->getNumVBases() > 0)) {
    return convertCXXRecordType(CXXRd);
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
QualTypeMapper::convertCXXRecordType(const CXXRecordDecl *RD) {
  const ASTRecordLayout &Layout = ASTCtx.getASTRecordLayout(RD);
  SmallVector<llvm::abi::FieldInfo, 16> Fields;

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

    Fields.emplace_back(BaseType, BaseOffset);
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

    Fields.emplace_back(VBaseType, VBaseOffset);
  }
  computeFieldInfo(RD, Fields, Layout);

  llvm::sort(Fields,
             [](const llvm::abi::FieldInfo &A, const llvm::abi::FieldInfo &B) {
               return A.OffsetInBits < B.OffsetInBits;
             });

  llvm::TypeSize Size =
      llvm::TypeSize::getFixed(Layout.getSize().getQuantity() * 8);
  llvm::Align Alignment = llvm::Align(Layout.getAlignment().getQuantity());

  return Builder.getStructType(Fields, Size, Alignment);
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
/// \return LLVM ABI StructType
const llvm::abi::StructType *
QualTypeMapper::convertStructType(const clang::RecordDecl *RD) {
  const ASTRecordLayout &Layout = ASTCtx.getASTRecordLayout(RD);

  SmallVector<llvm::abi::FieldInfo, 16> Fields;
  computeFieldInfo(RD, Fields, Layout);

  llvm::TypeSize Size =
      llvm::TypeSize::getFixed(Layout.getSize().getQuantity() * 8);
  llvm::Align Alignment = llvm::Align(Layout.getAlignment().getQuantity());

  return Builder.getStructType(Fields, Size, Alignment);
}

/// Converts C union types where all fields occupy the same memory location.
/// The union size is determined by its largest member, and all fields
/// start at offset 0.
///
/// \param RD The RecordDecl representing the union
/// \return LLVM ABI UnionType
const llvm::abi::UnionType *
QualTypeMapper::convertUnionType(const clang::RecordDecl *RD) {
  const ASTRecordLayout &Layout = ASTCtx.getASTRecordLayout(RD);

  SmallVector<llvm::abi::FieldInfo, 16> Fields;
  computeFieldInfo(RD, Fields, Layout);

  llvm::TypeSize Size =
      llvm::TypeSize::getFixed(Layout.getSize().getQuantity() * 8);
  llvm::Align Alignment = llvm::Align(Layout.getAlignment().getQuantity());

  return Builder.getUnionType(Fields, Size, Alignment);
}

llvm::Align QualTypeMapper::getTypeAlign(QualType QT) const {
  return llvm::Align(ASTCtx.getTypeAlign(QT));
}

const llvm::abi::Type *
QualTypeMapper::createPointerTypeForPointee(QualType PointeeType) {
  auto AddrSpace = PointeeType.getAddressSpace();
  auto PointerSize = ASTCtx.getTargetInfo().getPointerWidth(AddrSpace);
  llvm::Align Alignment =
      llvm::Align(ASTCtx.getTargetInfo().getPointerAlign(AddrSpace));
  return Builder.getPointerType(PointerSize, Alignment);
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

    if (IsBitField)
      BitFieldWidth = FD->getBitWidthValue();

    Fields.emplace_back(FieldType, OffsetInBits, IsBitField, BitFieldWidth);
    ++FieldIndex;
  }
}

} // namespace CodeGen
} // namespace clang
