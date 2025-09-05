//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "CIRGenBuilder.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "clang/CIR/MissingFeatures.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace clang::CIRGen;

mlir::Value CIRGenBuilderTy::maybeBuildArrayDecay(mlir::Location loc,
                                                  mlir::Value arrayPtr,
                                                  mlir::Type eltTy) {
  const auto arrayPtrTy = mlir::cast<cir::PointerType>(arrayPtr.getType());
  const auto arrayTy = mlir::dyn_cast<cir::ArrayType>(arrayPtrTy.getPointee());

  if (arrayTy) {
    const cir::PointerType flatPtrTy = getPointerTo(arrayTy.getElementType());
    return create<cir::CastOp>(loc, flatPtrTy, cir::CastKind::array_to_ptrdecay,
                               arrayPtr);
  }

  assert(arrayPtrTy.getPointee() == eltTy &&
         "flat pointee type must match original array element type");
  return arrayPtr;
}

mlir::Value CIRGenBuilderTy::getArrayElement(mlir::Location arrayLocBegin,
                                             mlir::Location arrayLocEnd,
                                             mlir::Value arrayPtr,
                                             mlir::Type eltTy, mlir::Value idx,
                                             bool shouldDecay) {
  mlir::Value basePtr = arrayPtr;
  if (shouldDecay)
    basePtr = maybeBuildArrayDecay(arrayLocBegin, arrayPtr, eltTy);
  const mlir::Type flatPtrTy = basePtr.getType();
  return create<cir::PtrStrideOp>(arrayLocEnd, flatPtrTy, basePtr, idx);
}

cir::ConstantOp CIRGenBuilderTy::getConstInt(mlir::Location loc,
                                             llvm::APSInt intVal) {
  bool isSigned = intVal.isSigned();
  unsigned width = intVal.getBitWidth();
  cir::IntType t = isSigned ? getSIntNTy(width) : getUIntNTy(width);
  return getConstInt(loc, t,
                     isSigned ? intVal.getSExtValue() : intVal.getZExtValue());
}

cir::ConstantOp CIRGenBuilderTy::getConstInt(mlir::Location loc,
                                             llvm::APInt intVal) {
  return getConstInt(loc, llvm::APSInt(intVal));
}

cir::ConstantOp CIRGenBuilderTy::getConstInt(mlir::Location loc, mlir::Type t,
                                             uint64_t c) {
  assert(mlir::isa<cir::IntType>(t) && "expected cir::IntType");
  return create<cir::ConstantOp>(loc, cir::IntAttr::get(t, c));
}

cir::ConstantOp
clang::CIRGen::CIRGenBuilderTy::getConstFP(mlir::Location loc, mlir::Type t,
                                           llvm::APFloat fpVal) {
  assert(mlir::isa<cir::FPTypeInterface>(t) && "expected floating point type");
  return create<cir::ConstantOp>(loc, cir::FPAttr::get(t, fpVal));
}

void CIRGenBuilderTy::computeGlobalViewIndicesFromFlatOffset(
    int64_t offset, mlir::Type ty, cir::CIRDataLayout layout,
    llvm::SmallVectorImpl<int64_t> &indices) {
  if (!offset)
    return;

  auto getIndexAndNewOffset =
      [](int64_t offset, int64_t eltSize) -> std::pair<int64_t, int64_t> {
    int64_t divRet = offset / eltSize;
    if (divRet < 0)
      divRet -= 1; // make sure offset is positive
    int64_t modRet = offset - (divRet * eltSize);
    return {divRet, modRet};
  };

  mlir::Type subType =
      llvm::TypeSwitch<mlir::Type, mlir::Type>(ty)
          .Case<cir::ArrayType>([&](auto arrayTy) {
            int64_t eltSize = layout.getTypeAllocSize(arrayTy.getElementType());
            const auto [index, newOffset] =
                getIndexAndNewOffset(offset, eltSize);
            indices.push_back(index);
            offset = newOffset;
            return arrayTy.getElementType();
          })
          .Case<cir::RecordType>([&](auto recordTy) {
            ArrayRef<mlir::Type> elts = recordTy.getMembers();
            int64_t pos = 0;
            for (size_t i = 0; i < elts.size(); ++i) {
              int64_t eltSize =
                  (int64_t)layout.getTypeAllocSize(elts[i]).getFixedValue();
              unsigned alignMask = layout.getABITypeAlign(elts[i]).value() - 1;
              if (recordTy.getPacked())
                alignMask = 0;
              // Union's fields have the same offset, so no need to change pos
              // here, we just need to find eltSize that is greater then the
              // required offset. The same is true for the similar union type
              // check below
              if (!recordTy.isUnion())
                pos = (pos + alignMask) & ~alignMask;
              assert(offset >= 0);
              if (offset < pos + eltSize) {
                indices.push_back(i);
                offset -= pos;
                return elts[i];
              }
              // No need to update pos here, see the comment above.
              if (!recordTy.isUnion())
                pos += eltSize;
            }
            llvm_unreachable("offset was not found within the record");
          })
          .Default([](mlir::Type otherTy) {
            llvm_unreachable("unexpected type");
            return otherTy; // Even though this is unreachable, we need to
                            // return a type to satisfy the return type of the
                            // lambda.
          });

  assert(subType);
  computeGlobalViewIndicesFromFlatOffset(offset, subType, layout, indices);
}

cir::RecordType clang::CIRGen::CIRGenBuilderTy::getCompleteRecordType(
    mlir::ArrayAttr fields, bool packed, bool padded, llvm::StringRef name) {
  assert(!cir::MissingFeatures::astRecordDeclAttr());
  llvm::SmallVector<mlir::Type> members;
  members.reserve(fields.size());
  llvm::transform(fields, std::back_inserter(members),
                  [](mlir::Attribute attr) {
                    return mlir::cast<mlir::TypedAttr>(attr).getType();
                  });

  if (name.empty())
    return getAnonRecordTy(members, packed, padded);

  return getCompleteNamedRecordType(members, packed, padded, name);
}

mlir::Attribute clang::CIRGen::CIRGenBuilderTy::getConstRecordOrZeroAttr(
    mlir::ArrayAttr arrayAttr, bool packed, bool padded, mlir::Type type) {
  auto recordTy = mlir::cast_or_null<cir::RecordType>(type);

  // Record type not specified: create anon record type from members.
  if (!recordTy) {
    recordTy = getCompleteRecordType(arrayAttr, packed, padded);
  }

  // Return zero or anonymous constant record.
  const bool isZero = llvm::all_of(
      arrayAttr, [&](mlir::Attribute a) { return isNullValue(a); });
  if (isZero)
    return cir::ZeroAttr::get(recordTy);
  return cir::ConstRecordAttr::get(recordTy, arrayAttr);
}

// This can't be defined in Address.h because that file is included by
// CIRGenBuilder.h
Address Address::withElementType(CIRGenBuilderTy &builder,
                                 mlir::Type elemTy) const {
  assert(!cir::MissingFeatures::addressOffset());
  assert(!cir::MissingFeatures::addressIsKnownNonNull());
  assert(!cir::MissingFeatures::addressPointerAuthInfo());

  return Address(builder.createPtrBitcast(getBasePointer(), elemTy), elemTy,
                 getAlignment());
}
