#include "clang/AST/RecordLayout.h"
#include "clang/AST/Type.h"
#include "clang/Analysis/Analyses/ThreadSafetyTIL.h"
#include "clang/Basic/LLVM.h"
#include "clang/Basic/TargetInfo.h"
#include "llvm/ABI/Types.h"
#include "llvm/Support/Alignment.h"
#include "llvm/Support/Casting.h"
#include <clang/ABI/QualTypeMapper.h>

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
  switch (BT->getKind()) {
  case BuiltinType::Void:
    return Builder.getVoidType();

  case BuiltinType::Bool:
  case BuiltinType::UChar:
  case BuiltinType::Char_U:
  case BuiltinType::UShort:
    return Builder.getIntegerType(ASTCtx.getTypeSize(QualType(BT, 0)),
                                  getTypeAlign(QualType(BT, 0)), false);

  case BuiltinType::Char_S:
  case BuiltinType::SChar:
  case BuiltinType::Short:
    return Builder.getIntegerType(ASTCtx.getCharWidth(),
                                  getTypeAlign(QualType(BT, 0)), true);

  case BuiltinType::WChar_U:
    return Builder.getIntegerType(ASTCtx.getCharWidth(),
                                  getTypeAlign(QualType(BT, 0)), false);

  case BuiltinType::WChar_S:
    return Builder.getIntegerType(ASTCtx.getCharWidth(),
                                  getTypeAlign(QualType(BT, 0)), true);

  case BuiltinType::Char8:
    return Builder.getIntegerType(8, getTypeAlign(QualType(BT, 0)), false);

  case BuiltinType::Char16:
    return Builder.getIntegerType(16, getTypeAlign(QualType(BT, 0)), false);

  case BuiltinType::Char32:
    return Builder.getIntegerType(32, getTypeAlign(QualType(BT, 0)), false);

  case BuiltinType::Int:
  case BuiltinType::UInt:
    return Builder.getIntegerType(ASTCtx.getIntWidth(QualType(BT, 0)),
                                  getTypeAlign(QualType(BT, 0)),
                                  BT->getKind() == BuiltinType::Int);

  case BuiltinType::Long:
  case BuiltinType::ULong:
    return Builder.getIntegerType(ASTCtx.getTypeSize(QualType(BT, 0)),
                                  getTypeAlign(QualType(BT, 0)),
                                  BT->getKind() == BuiltinType::Long);

  case BuiltinType::LongLong:
  case BuiltinType::ULongLong:
    return Builder.getIntegerType(ASTCtx.getTypeSize(QualType(BT, 0)),
                                  getTypeAlign(QualType(BT, 0)),
                                  BT->getKind() == BuiltinType::LongLong);

  case BuiltinType::Int128:
  case BuiltinType::UInt128:
    return Builder.getIntegerType(128, getTypeAlign(QualType(BT, 0)),
                                  BT->getKind() == BuiltinType::Int128);

  case BuiltinType::Half:
  case BuiltinType::Float16:
    return Builder.getFloatType(llvm::APFloat::IEEEhalf(),
                                getTypeAlign(QualType(BT, 0)));

  case BuiltinType::Float:
    return Builder.getFloatType(llvm::APFloat::IEEEsingle(),
                                getTypeAlign(QualType(BT, 0)));

  case BuiltinType::Double:
    return Builder.getFloatType(llvm::APFloat::IEEEdouble(),
                                getTypeAlign(QualType(BT, 0)));

  case BuiltinType::LongDouble:
    return Builder.getFloatType(ASTCtx.getFloatTypeSemantics(QualType(BT, 0)),
                                getTypeAlign(QualType(BT, 0)));

  case BuiltinType::BFloat16:
    return Builder.getFloatType(llvm::APFloat::BFloat(),
                                getTypeAlign(QualType(BT, 0)));

  case BuiltinType::Float128:
    return Builder.getFloatType(llvm::APFloat::IEEEquad(),
                                getTypeAlign(QualType(BT, 0)));

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
