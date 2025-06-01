//==---- QualtypeMapper.cpp - Maps Clang Qualtype to LLVMABI Types ---------==//
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
#include "clang/CodeGen/QualtypeMapper.h"
#include "clang/AST/Decl.h"
#include "clang/AST/RecordLayout.h"
#include "clang/AST/Type.h"
#include "clang/Basic/LLVM.h"
#include "clang/Basic/TargetInfo.h"
#include "llvm/ABI/Types.h"
#include "llvm/Support/Alignment.h"
#include "llvm/Support/TypeSize.h"

namespace clang {
namespace mapper {

const llvm::abi::Type *QualTypeMapper::convertType(QualType QT) {
  QT = QT.getCanonicalType().getUnqualifiedType();

  auto It = TypeCache.find(QT);
  if (It != TypeCache.end())
    return It->second;

  const llvm::abi::Type *Result = nullptr;
  if (const auto *BT = dyn_cast<BuiltinType>(QT.getTypePtr())) {
    Result = convertBuiltinType(BT);
  } else if (const auto *PT = dyn_cast<PointerType>(QT.getTypePtr())) {
    Result = convertPointerType(PT);
  } else if (const auto *AT = dyn_cast<ArrayType>(QT.getTypePtr())) {
    Result = convertArrayType(AT);
  } else if (const auto *VT = dyn_cast<VectorType>(QT.getTypePtr())) {
    Result = convertVectorType(VT);
  } else if (const auto *RT = dyn_cast<RecordType>(QT.getTypePtr())) {
    Result = convertRecordType(RT);
  } else if (const auto *ET = dyn_cast<EnumType>(QT.getTypePtr())) {
    Result = convertEnumType(ET);
  } else {
    // TODO: Write Fallback logic for unsupported types.
  }
  TypeCache[QT] = Result;
  return Result;
}

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
    return Builder.getIntegerType(ASTCtx.getTypeSize(QualType(BT, 0)),
                                  getTypeAlign(QualType(BT, 0)), false);
  }
}

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
  // TODO: This of a better fallback.
  return Builder.getArrayType(ElementType, 1);
}

const llvm::abi::Type *QualTypeMapper::convertVectorType(const VectorType *VT) {
  const llvm::abi::Type *ElementType = convertType(VT->getElementType());
  uint64_t NumElements = VT->getNumElements();

  llvm::Align VectorAlign = getTypeAlign(QualType(VT, 0));

  return Builder.getVectorType(ElementType, NumElements, VectorAlign);
}

const llvm::abi::Type *QualTypeMapper::convertRecordType(const RecordType *RT) {
  const RecordDecl *RD = RT->getDecl()->getDefinition();
  if (!RD) {
    SmallVector<llvm::abi::FieldInfo, 0> Fields;
    return Builder.getStructType(Fields, llvm::TypeSize::getFixed(0),
                                 llvm::Align(1));
  }

  if (RD->isUnion())
    return convertUnionType(RD);
  return convertStructType(RD);
}

const llvm::abi::Type *
QualTypeMapper::convertPointerType(const clang::PointerType *PT) {
  return createPointerTypeForPointee(PT->getPointeeType());
}

const llvm::abi::Type *
QualTypeMapper::convertEnumType(const clang::EnumType *ET) {
  const EnumDecl *ED = ET->getDecl();
  QualType UnderlyingType = ED->getIntegerType();

  if (UnderlyingType.isNull())
    UnderlyingType = ASTCtx.IntTy;

  return convertType(UnderlyingType);
}

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

void QualTypeMapper::computeFieldInfo(
    const RecordDecl *RD, SmallVectorImpl<llvm::abi::FieldInfo> &Fields,
    const ASTRecordLayout &Layout) {
  unsigned FieldIndex = 0;

  for (const auto *FD : RD->fields()) {
    const llvm::abi::Type *FieldType = convertType(FD->getType());
    uint64_t OffsetInBits = Layout.getFieldOffset(FieldIndex);

    bool IsBitField = FD->isBitField();
    uint64_t BitFieldWidth = 0;

    if (IsBitField) {
      BitFieldWidth = FD->getBitWidthValue();
    }

    Fields.emplace_back(FieldType, OffsetInBits, IsBitField, BitFieldWidth);
    ++FieldIndex;
  }
}

} // namespace mapper
} // namespace clang
