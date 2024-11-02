//===-- Character.cpp -----------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Coding style: https://mlir.llvm.org/getting_started/DeveloperGuide/
//
//===----------------------------------------------------------------------===//

#include "flang/Optimizer/Builder/Character.h"
#include "flang/Lower/Todo.h"
#include "flang/Optimizer/Builder/DoLoopHelper.h"
#include "flang/Optimizer/Builder/FIRBuilder.h"
#include "llvm/Support/Debug.h"
#include <optional>

#define DEBUG_TYPE "flang-lower-character"

//===----------------------------------------------------------------------===//
// CharacterExprHelper implementation
//===----------------------------------------------------------------------===//

/// Unwrap all the ref and box types and return the inner element type.
static mlir::Type unwrapBoxAndRef(mlir::Type type) {
  if (auto boxType = type.dyn_cast<fir::BoxCharType>())
    return boxType.getEleTy();
  while (true) {
    type = fir::unwrapRefType(type);
    if (auto boxTy = type.dyn_cast<fir::BoxType>())
      type = boxTy.getEleTy();
    else
      break;
  }
  return type;
}

/// Unwrap base fir.char<kind,len> type.
static fir::CharacterType recoverCharacterType(mlir::Type type) {
  type = fir::unwrapSequenceType(unwrapBoxAndRef(type));
  if (auto charTy = type.dyn_cast<fir::CharacterType>())
    return charTy;
  llvm::report_fatal_error("expected a character type");
}

bool fir::factory::CharacterExprHelper::isCharacterScalar(mlir::Type type) {
  type = unwrapBoxAndRef(type);
  return !type.isa<fir::SequenceType>() && fir::isa_char(type);
}

bool fir::factory::CharacterExprHelper::isArray(mlir::Type type) {
  type = unwrapBoxAndRef(type);
  if (auto seqTy = type.dyn_cast<fir::SequenceType>())
    return fir::isa_char(seqTy.getEleTy());
  return false;
}

fir::CharacterType
fir::factory::CharacterExprHelper::getCharacterType(mlir::Type type) {
  assert(isCharacterScalar(type) && "expected scalar character");
  return recoverCharacterType(type);
}

fir::CharacterType
fir::factory::CharacterExprHelper::getCharType(mlir::Type type) {
  return recoverCharacterType(type);
}

fir::CharacterType fir::factory::CharacterExprHelper::getCharacterType(
    const fir::CharBoxValue &box) {
  return getCharacterType(box.getBuffer().getType());
}

fir::CharacterType
fir::factory::CharacterExprHelper::getCharacterType(mlir::Value str) {
  return getCharacterType(str.getType());
}

/// Determine the static size of the character. Returns the computed size, not
/// an IR Value.
static std::optional<fir::CharacterType::LenType>
getCompileTimeLength(const fir::CharBoxValue &box) {
  auto len = recoverCharacterType(box.getBuffer().getType()).getLen();
  if (len == fir::CharacterType::unknownLen())
    return {};
  return len;
}

/// Detect the precondition that the value `str` does not reside in memory. Such
/// values will have a type `!fir.array<...x!fir.char<N>>` or `!fir.char<N>`.
LLVM_ATTRIBUTE_UNUSED static bool needToMaterialize(mlir::Value str) {
  return str.getType().isa<fir::SequenceType>() || fir::isa_char(str.getType());
}

/// This is called only if `str` does not reside in memory. Such a bare string
/// value will be converted into a memory-based temporary and an extended
/// boxchar value returned.
fir::CharBoxValue
fir::factory::CharacterExprHelper::materializeValue(mlir::Value str) {
  assert(needToMaterialize(str));
  auto ty = str.getType();
  assert(isCharacterScalar(ty) && "expected scalar character");
  auto charTy = ty.dyn_cast<fir::CharacterType>();
  if (!charTy || charTy.getLen() == fir::CharacterType::unknownLen()) {
    LLVM_DEBUG(llvm::dbgs() << "cannot materialize: " << str << '\n');
    llvm_unreachable("must be a !fir.char<N> type");
  }
  auto len = builder.createIntegerConstant(
      loc, builder.getCharacterLengthType(), charTy.getLen());
  auto temp = builder.create<fir::AllocaOp>(loc, charTy);
  builder.create<fir::StoreOp>(loc, str, temp);
  LLVM_DEBUG(llvm::dbgs() << "materialized as local: " << str << " -> (" << temp
                          << ", " << len << ")\n");
  return {temp, len};
}

fir::ExtendedValue
fir::factory::CharacterExprHelper::toExtendedValue(mlir::Value character,
                                                   mlir::Value len) {
  auto lenType = builder.getCharacterLengthType();
  auto type = character.getType();
  auto base = fir::isa_passbyref_type(type) ? character : mlir::Value{};
  auto resultLen = len;
  llvm::SmallVector<mlir::Value> extents;

  if (auto eleType = fir::dyn_cast_ptrEleTy(type))
    type = eleType;

  if (auto arrayType = type.dyn_cast<fir::SequenceType>()) {
    type = arrayType.getEleTy();
    auto indexType = builder.getIndexType();
    for (auto extent : arrayType.getShape()) {
      if (extent == fir::SequenceType::getUnknownExtent())
        break;
      extents.emplace_back(
          builder.createIntegerConstant(loc, indexType, extent));
    }
    // Last extent might be missing in case of assumed-size. If more extents
    // could not be deduced from type, that's an error (a fir.box should
    // have been used in the interface).
    if (extents.size() + 1 < arrayType.getShape().size())
      mlir::emitError(loc, "cannot retrieve array extents from type");
  }

  if (auto charTy = type.dyn_cast<fir::CharacterType>()) {
    if (!resultLen && charTy.getLen() != fir::CharacterType::unknownLen())
      resultLen = builder.createIntegerConstant(loc, lenType, charTy.getLen());
  } else if (auto boxCharType = type.dyn_cast<fir::BoxCharType>()) {
    auto refType = builder.getRefType(boxCharType.getEleTy());
    // If the embox is accessible, use its operand to avoid filling
    // the generated fir with embox/unbox.
    mlir::Value boxCharLen;
    if (auto definingOp = character.getDefiningOp()) {
      if (auto box = mlir::dyn_cast<fir::EmboxCharOp>(definingOp)) {
        base = box.getMemref();
        boxCharLen = box.getLen();
      }
    }
    if (!boxCharLen) {
      auto unboxed =
          builder.create<fir::UnboxCharOp>(loc, refType, lenType, character);
      base = builder.createConvert(loc, refType, unboxed.getResult(0));
      boxCharLen = unboxed.getResult(1);
    }
    if (!resultLen) {
      resultLen = boxCharLen;
    }
  } else if (type.isa<fir::BoxType>()) {
    mlir::emitError(loc, "descriptor or derived type not yet handled");
  } else {
    llvm_unreachable("Cannot translate mlir::Value to character ExtendedValue");
  }

  if (!base) {
    if (auto load =
            mlir::dyn_cast_or_null<fir::LoadOp>(character.getDefiningOp())) {
      base = load.getOperand();
    } else {
      return materializeValue(fir::getBase(character));
    }
  }
  if (!resultLen)
    llvm::report_fatal_error("no dynamic length found for character");
  if (!extents.empty())
    return fir::CharArrayBoxValue{base, resultLen, extents};
  return fir::CharBoxValue{base, resultLen};
}

static mlir::Type getSingletonCharType(mlir::MLIRContext *ctxt, int kind) {
  return fir::CharacterType::getSingleton(ctxt, kind);
}

mlir::Value
fir::factory::CharacterExprHelper::createEmbox(const fir::CharBoxValue &box) {
  // Base CharBoxValue of CharArrayBoxValue are ok here (do not require a scalar
  // type)
  auto charTy = recoverCharacterType(box.getBuffer().getType());
  auto boxCharType =
      fir::BoxCharType::get(builder.getContext(), charTy.getFKind());
  auto refType = fir::ReferenceType::get(boxCharType.getEleTy());
  mlir::Value buff = box.getBuffer();
  // fir.boxchar requires a memory reference. Allocate temp if the character is
  // not in memory.
  if (!fir::isa_ref_type(buff.getType())) {
    auto temp = builder.createTemporary(loc, buff.getType());
    builder.create<fir::StoreOp>(loc, buff, temp);
    buff = temp;
  }
  buff = builder.createConvert(loc, refType, buff);
  // Convert in case the provided length is not of the integer type that must
  // be used in boxchar.
  auto len = builder.createConvert(loc, builder.getCharacterLengthType(),
                                   box.getLen());
  return builder.create<fir::EmboxCharOp>(loc, boxCharType, buff, len);
}

fir::CharBoxValue fir::factory::CharacterExprHelper::toScalarCharacter(
    const fir::CharArrayBoxValue &box) {
  if (box.getBuffer().getType().isa<fir::PointerType>())
    TODO(loc, "concatenating non contiguous character array into a scalar");

  // TODO: add a fast path multiplying new length at compile time if the info is
  // in the array type.
  auto lenType = builder.getCharacterLengthType();
  auto len = builder.createConvert(loc, lenType, box.getLen());
  for (auto extent : box.getExtents())
    len = builder.create<mlir::arith::MulIOp>(
        loc, len, builder.createConvert(loc, lenType, extent));

  // TODO: typeLen can be improved in compiled constant cases
  // TODO: allow bare fir.array<> (no ref) conversion here ?
  auto typeLen = fir::CharacterType::unknownLen();
  auto kind = recoverCharacterType(box.getBuffer().getType()).getFKind();
  auto charTy = fir::CharacterType::get(builder.getContext(), kind, typeLen);
  auto type = fir::ReferenceType::get(charTy);
  auto buffer = builder.createConvert(loc, type, box.getBuffer());
  return {buffer, len};
}

mlir::Value fir::factory::CharacterExprHelper::createEmbox(
    const fir::CharArrayBoxValue &box) {
  // Use same embox as for scalar. It's losing the actual data size information
  // (We do not multiply the length by the array size), but that is what Fortran
  // call interfaces using boxchar expect.
  return createEmbox(static_cast<const fir::CharBoxValue &>(box));
}

/// Get the address of the element at position \p index of the scalar character
/// \p buffer.
/// \p buffer must be of type !fir.ref<fir.char<k, len>>. The length may be
/// unknown. \p index must have any integer type, and is zero based. The return
/// value is a singleton address (!fir.ref<!fir.char<kind>>)
mlir::Value
fir::factory::CharacterExprHelper::createElementAddr(mlir::Value buffer,
                                                     mlir::Value index) {
  // The only way to address an element of a fir.ref<char<kind, len>> is to cast
  // it to a fir.array<len x fir.char<kind>> and use fir.coordinate_of.
  auto bufferType = buffer.getType();
  assert(fir::isa_ref_type(bufferType));
  assert(isCharacterScalar(bufferType));
  auto charTy = recoverCharacterType(bufferType);
  auto singleTy = getSingletonCharType(builder.getContext(), charTy.getFKind());
  auto singleRefTy = builder.getRefType(singleTy);
  auto extent = fir::SequenceType::getUnknownExtent();
  if (charTy.getLen() != fir::CharacterType::unknownLen())
    extent = charTy.getLen();
  auto coorTy = builder.getRefType(fir::SequenceType::get({extent}, singleTy));

  auto coor = builder.createConvert(loc, coorTy, buffer);
  auto i = builder.createConvert(loc, builder.getIndexType(), index);
  return builder.create<fir::CoordinateOp>(loc, singleRefTy, coor, i);
}

/// Load a character out of `buff` from offset `index`.
/// `buff` must be a reference to memory.
mlir::Value
fir::factory::CharacterExprHelper::createLoadCharAt(mlir::Value buff,
                                                    mlir::Value index) {
  LLVM_DEBUG(llvm::dbgs() << "load a char: " << buff << " type: "
                          << buff.getType() << " at: " << index << '\n');
  return builder.create<fir::LoadOp>(loc, createElementAddr(buff, index));
}

/// Store the singleton character `c` to `str` at offset `index`.
/// `str` must be a reference to memory.
void fir::factory::CharacterExprHelper::createStoreCharAt(mlir::Value str,
                                                          mlir::Value index,
                                                          mlir::Value c) {
  LLVM_DEBUG(llvm::dbgs() << "store the char: " << c << " into: " << str
                          << " type: " << str.getType() << " at: " << index
                          << '\n');
  auto addr = createElementAddr(str, index);
  builder.create<fir::StoreOp>(loc, c, addr);
}

// FIXME: this temp is useless... either fir.coordinate_of needs to
// work on "loaded" characters (!fir.array<len x fir.char<kind>>) or
// character should never be loaded.
// If this is a fir.array<>, allocate and store the value so that
// fir.cooridnate_of can be use on the value.
mlir::Value fir::factory::CharacterExprHelper::getCharBoxBuffer(
    const fir::CharBoxValue &box) {
  auto buff = box.getBuffer();
  if (fir::isa_char(buff.getType())) {
    auto newBuff = builder.create<fir::AllocaOp>(loc, buff.getType());
    builder.create<fir::StoreOp>(loc, buff, newBuff);
    return newBuff;
  }
  return buff;
}

/// Create a loop to copy `count` characters from `src` to `dest`. Note that the
/// KIND indicates the number of bits in a code point. (ASCII, UCS-2, or UCS-4.)
void fir::factory::CharacterExprHelper::createCopy(
    const fir::CharBoxValue &dest, const fir::CharBoxValue &src,
    mlir::Value count) {
  auto fromBuff = getCharBoxBuffer(src);
  auto toBuff = getCharBoxBuffer(dest);
  LLVM_DEBUG(llvm::dbgs() << "create char copy from: "; src.dump();
             llvm::dbgs() << " to: "; dest.dump();
             llvm::dbgs() << " count: " << count << '\n');
  auto kind = getCharacterKind(src.getBuffer().getType());
  // If the src and dest are the same KIND, then use memmove to move the bits.
  // We don't have to worry about overlapping ranges with memmove.
  if (getCharacterKind(dest.getBuffer().getType()) == kind) {
    auto bytes = builder.getKindMap().getCharacterBitsize(kind) / 8;
    auto i64Ty = builder.getI64Type();
    auto kindBytes = builder.createIntegerConstant(loc, i64Ty, bytes);
    auto castCount = builder.createConvert(loc, i64Ty, count);
    auto totalBytes =
        builder.create<mlir::arith::MulIOp>(loc, kindBytes, castCount);
    auto notVolatile = builder.createBool(loc, false);
    auto memmv = getLlvmMemmove(builder);
    auto argTys = memmv.getFunctionType().getInputs();
    auto toPtr = builder.createConvert(loc, argTys[0], toBuff);
    auto fromPtr = builder.createConvert(loc, argTys[1], fromBuff);
    builder.create<fir::CallOp>(
        loc, memmv, mlir::ValueRange{toPtr, fromPtr, totalBytes, notVolatile});
    return;
  }

  // Convert a CHARACTER of one KIND into a CHARACTER of another KIND.
  builder.create<fir::CharConvertOp>(loc, src.getBuffer(), count,
                                     dest.getBuffer());
}

void fir::factory::CharacterExprHelper::createPadding(
    const fir::CharBoxValue &str, mlir::Value lower, mlir::Value upper) {
  auto blank = createBlankConstant(getCharacterType(str));
  // Always create the loop, if upper < lower, no iteration will be
  // executed.
  auto toBuff = getCharBoxBuffer(str);
  fir::factory::DoLoopHelper{builder, loc}.createLoop(
      lower, upper, [&](fir::FirOpBuilder &, mlir::Value index) {
        createStoreCharAt(toBuff, index, blank);
      });
}

fir::CharBoxValue
fir::factory::CharacterExprHelper::createCharacterTemp(mlir::Type type,
                                                       mlir::Value len) {
  auto kind = recoverCharacterType(type).getFKind();
  auto typeLen = fir::CharacterType::unknownLen();
  // If len is a constant, reflect the length in the type.
  if (auto cstLen = getIntIfConstant(len))
    typeLen = *cstLen;
  auto *ctxt = builder.getContext();
  auto charTy = fir::CharacterType::get(ctxt, kind, typeLen);
  llvm::SmallVector<mlir::Value> lenParams;
  if (typeLen == fir::CharacterType::unknownLen())
    lenParams.push_back(len);
  auto ref = builder.allocateLocal(loc, charTy, "", ".chrtmp",
                                   /*shape=*/llvm::None, lenParams);
  return {ref, len};
}

fir::CharBoxValue fir::factory::CharacterExprHelper::createTempFrom(
    const fir::ExtendedValue &source) {
  const auto *charBox = source.getCharBox();
  if (!charBox)
    fir::emitFatalError(loc, "source must be a fir::CharBoxValue");
  auto len = charBox->getLen();
  auto sourceTy = charBox->getBuffer().getType();
  auto temp = createCharacterTemp(sourceTy, len);
  if (fir::isa_ref_type(sourceTy)) {
    createCopy(temp, *charBox, len);
  } else {
    auto ref = builder.createConvert(loc, builder.getRefType(sourceTy),
                                     temp.getBuffer());
    builder.create<fir::StoreOp>(loc, charBox->getBuffer(), ref);
  }
  return temp;
}

// Simple length one character assignment without loops.
void fir::factory::CharacterExprHelper::createLengthOneAssign(
    const fir::CharBoxValue &lhs, const fir::CharBoxValue &rhs) {
  auto addr = lhs.getBuffer();
  auto toTy = fir::unwrapRefType(addr.getType());
  mlir::Value val = rhs.getBuffer();
  if (fir::isa_ref_type(val.getType()))
    val = builder.create<fir::LoadOp>(loc, val);
  val = builder.createConvert(loc, toTy, val);
  builder.create<fir::StoreOp>(loc, val, addr);
}

/// Returns the minimum of integer mlir::Value \p a and \b.
mlir::Value genMin(fir::FirOpBuilder &builder, mlir::Location loc,
                   mlir::Value a, mlir::Value b) {
  auto cmp = builder.create<mlir::arith::CmpIOp>(
      loc, mlir::arith::CmpIPredicate::slt, a, b);
  return builder.create<mlir::arith::SelectOp>(loc, cmp, a, b);
}

void fir::factory::CharacterExprHelper::createAssign(
    const fir::CharBoxValue &lhs, const fir::CharBoxValue &rhs) {
  auto rhsCstLen = getCompileTimeLength(rhs);
  auto lhsCstLen = getCompileTimeLength(lhs);
  bool compileTimeSameLength =
      lhsCstLen && rhsCstLen && *lhsCstLen == *rhsCstLen;

  if (compileTimeSameLength && *lhsCstLen == 1) {
    createLengthOneAssign(lhs, rhs);
    return;
  }

  // Copy the minimum of the lhs and rhs lengths and pad the lhs remainder
  // if needed.
  auto copyCount = lhs.getLen();
  auto idxTy = builder.getIndexType();
  if (!compileTimeSameLength) {
    auto lhsLen = builder.createConvert(loc, idxTy, lhs.getLen());
    auto rhsLen = builder.createConvert(loc, idxTy, rhs.getLen());
    copyCount = genMin(builder, loc, lhsLen, rhsLen);
  }

  // Actual copy
  createCopy(lhs, rhs, copyCount);

  // Pad if needed.
  if (!compileTimeSameLength) {
    auto one = builder.createIntegerConstant(loc, lhs.getLen().getType(), 1);
    auto maxPadding =
        builder.create<mlir::arith::SubIOp>(loc, lhs.getLen(), one);
    createPadding(lhs, copyCount, maxPadding);
  }
}

fir::CharBoxValue fir::factory::CharacterExprHelper::createConcatenate(
    const fir::CharBoxValue &lhs, const fir::CharBoxValue &rhs) {
  auto lhsLen = builder.createConvert(loc, builder.getCharacterLengthType(),
                                      lhs.getLen());
  auto rhsLen = builder.createConvert(loc, builder.getCharacterLengthType(),
                                      rhs.getLen());
  mlir::Value len = builder.create<mlir::arith::AddIOp>(loc, lhsLen, rhsLen);
  auto temp = createCharacterTemp(getCharacterType(rhs), len);
  createCopy(temp, lhs, lhsLen);
  auto one = builder.createIntegerConstant(loc, len.getType(), 1);
  auto upperBound = builder.create<mlir::arith::SubIOp>(loc, len, one);
  auto lhsLenIdx = builder.createConvert(loc, builder.getIndexType(), lhsLen);
  auto fromBuff = getCharBoxBuffer(rhs);
  auto toBuff = getCharBoxBuffer(temp);
  fir::factory::DoLoopHelper{builder, loc}.createLoop(
      lhsLenIdx, upperBound, one,
      [&](fir::FirOpBuilder &bldr, mlir::Value index) {
        auto rhsIndex = bldr.create<mlir::arith::SubIOp>(loc, index, lhsLenIdx);
        auto charVal = createLoadCharAt(fromBuff, rhsIndex);
        createStoreCharAt(toBuff, index, charVal);
      });
  return temp;
}

fir::CharBoxValue fir::factory::CharacterExprHelper::createSubstring(
    const fir::CharBoxValue &box, llvm::ArrayRef<mlir::Value> bounds) {
  // Constant need to be materialize in memory to use fir.coordinate_of.
  auto nbounds = bounds.size();
  if (nbounds < 1 || nbounds > 2) {
    mlir::emitError(loc, "Incorrect number of bounds in substring");
    return {mlir::Value{}, mlir::Value{}};
  }
  mlir::SmallVector<mlir::Value> castBounds;
  // Convert bounds to length type to do safe arithmetic on it.
  for (auto bound : bounds)
    castBounds.push_back(
        builder.createConvert(loc, builder.getCharacterLengthType(), bound));
  auto lowerBound = castBounds[0];
  // FIR CoordinateOp is zero based but Fortran substring are one based.
  auto one = builder.createIntegerConstant(loc, lowerBound.getType(), 1);
  auto offset =
      builder.create<mlir::arith::SubIOp>(loc, lowerBound, one).getResult();
  auto addr = createElementAddr(box.getBuffer(), offset);
  auto kind = getCharacterKind(box.getBuffer().getType());
  auto charTy = fir::CharacterType::getUnknownLen(builder.getContext(), kind);
  auto resultType = builder.getRefType(charTy);
  auto substringRef = builder.createConvert(loc, resultType, addr);

  // Compute the length.
  mlir::Value substringLen;
  if (nbounds < 2) {
    substringLen =
        builder.create<mlir::arith::SubIOp>(loc, box.getLen(), castBounds[0]);
  } else {
    substringLen =
        builder.create<mlir::arith::SubIOp>(loc, castBounds[1], castBounds[0]);
  }
  substringLen = builder.create<mlir::arith::AddIOp>(loc, substringLen, one);

  // Set length to zero if bounds were reversed (Fortran 2018 9.4.1)
  auto zero = builder.createIntegerConstant(loc, substringLen.getType(), 0);
  auto cdt = builder.create<mlir::arith::CmpIOp>(
      loc, mlir::arith::CmpIPredicate::slt, substringLen, zero);
  substringLen =
      builder.create<mlir::arith::SelectOp>(loc, cdt, zero, substringLen);

  return {substringRef, substringLen};
}

mlir::Value
fir::factory::CharacterExprHelper::createLenTrim(const fir::CharBoxValue &str) {
  // Note: Runtime for LEN_TRIM should also be available at some
  // point. For now use an inlined implementation.
  auto indexType = builder.getIndexType();
  auto len = builder.createConvert(loc, indexType, str.getLen());
  auto one = builder.createIntegerConstant(loc, indexType, 1);
  auto minusOne = builder.createIntegerConstant(loc, indexType, -1);
  auto zero = builder.createIntegerConstant(loc, indexType, 0);
  auto trueVal = builder.createIntegerConstant(loc, builder.getI1Type(), 1);
  auto blank = createBlankConstantCode(getCharacterType(str));
  mlir::Value lastChar = builder.create<mlir::arith::SubIOp>(loc, len, one);

  auto iterWhile =
      builder.create<fir::IterWhileOp>(loc, lastChar, zero, minusOne, trueVal,
                                       /*returnFinalCount=*/false, lastChar);
  auto insPt = builder.saveInsertionPoint();
  builder.setInsertionPointToStart(iterWhile.getBody());
  auto index = iterWhile.getInductionVar();
  // Look for first non-blank from the right of the character.
  auto fromBuff = getCharBoxBuffer(str);
  auto elemAddr = createElementAddr(fromBuff, index);
  auto codeAddr =
      builder.createConvert(loc, builder.getRefType(blank.getType()), elemAddr);
  auto c = builder.create<fir::LoadOp>(loc, codeAddr);
  auto isBlank = builder.create<mlir::arith::CmpIOp>(
      loc, mlir::arith::CmpIPredicate::eq, blank, c);
  llvm::SmallVector<mlir::Value> results = {isBlank, index};
  builder.create<fir::ResultOp>(loc, results);
  builder.restoreInsertionPoint(insPt);
  // Compute length after iteration (zero if all blanks)
  mlir::Value newLen =
      builder.create<mlir::arith::AddIOp>(loc, iterWhile.getResult(1), one);
  auto result = builder.create<mlir::arith::SelectOp>(
      loc, iterWhile.getResult(0), zero, newLen);
  return builder.createConvert(loc, builder.getCharacterLengthType(), result);
}

fir::CharBoxValue
fir::factory::CharacterExprHelper::createCharacterTemp(mlir::Type type,
                                                       int len) {
  assert(len >= 0 && "expected positive length");
  auto kind = recoverCharacterType(type).getFKind();
  auto charType = fir::CharacterType::get(builder.getContext(), kind, len);
  auto addr = builder.create<fir::AllocaOp>(loc, charType);
  auto mlirLen =
      builder.createIntegerConstant(loc, builder.getCharacterLengthType(), len);
  return {addr, mlirLen};
}

// Returns integer with code for blank. The integer has the same
// size as the character. Blank has ascii space code for all kinds.
mlir::Value fir::factory::CharacterExprHelper::createBlankConstantCode(
    fir::CharacterType type) {
  auto bits = builder.getKindMap().getCharacterBitsize(type.getFKind());
  auto intType = builder.getIntegerType(bits);
  return builder.createIntegerConstant(loc, intType, ' ');
}

mlir::Value fir::factory::CharacterExprHelper::createBlankConstant(
    fir::CharacterType type) {
  return createSingletonFromCode(createBlankConstantCode(type),
                                 type.getFKind());
}

void fir::factory::CharacterExprHelper::createAssign(
    const fir::ExtendedValue &lhs, const fir::ExtendedValue &rhs) {
  if (auto *str = rhs.getBoxOf<fir::CharBoxValue>()) {
    if (auto *to = lhs.getBoxOf<fir::CharBoxValue>()) {
      createAssign(*to, *str);
      return;
    }
  }
  TODO(loc, "character array assignment");
  // Note that it is not sure the array aspect should be handled
  // by this utility.
}

mlir::Value
fir::factory::CharacterExprHelper::createEmboxChar(mlir::Value addr,
                                                   mlir::Value len) {
  return createEmbox(fir::CharBoxValue{addr, len});
}

std::pair<mlir::Value, mlir::Value>
fir::factory::CharacterExprHelper::createUnboxChar(mlir::Value boxChar) {
  using T = std::pair<mlir::Value, mlir::Value>;
  return toExtendedValue(boxChar).match(
      [](const fir::CharBoxValue &b) -> T {
        return {b.getBuffer(), b.getLen()};
      },
      [](const fir::CharArrayBoxValue &b) -> T {
        return {b.getBuffer(), b.getLen()};
      },
      [](const auto &) -> T { llvm::report_fatal_error("not a character"); });
}

bool fir::factory::CharacterExprHelper::isCharacterLiteral(mlir::Type type) {
  if (auto seqType = type.dyn_cast<fir::SequenceType>())
    return (seqType.getShape().size() == 1) &&
           fir::isa_char(seqType.getEleTy());
  return false;
}

fir::KindTy
fir::factory::CharacterExprHelper::getCharacterKind(mlir::Type type) {
  assert(isCharacterScalar(type) && "expected scalar character");
  return recoverCharacterType(type).getFKind();
}

fir::KindTy
fir::factory::CharacterExprHelper::getCharacterOrSequenceKind(mlir::Type type) {
  return recoverCharacterType(type).getFKind();
}

bool fir::factory::CharacterExprHelper::hasConstantLengthInType(
    const fir::ExtendedValue &exv) {
  auto charTy = recoverCharacterType(fir::getBase(exv).getType());
  return charTy.hasConstantLen();
}

mlir::Value
fir::factory::CharacterExprHelper::createSingletonFromCode(mlir::Value code,
                                                           int kind) {
  auto charType = fir::CharacterType::get(builder.getContext(), kind, 1);
  auto bits = builder.getKindMap().getCharacterBitsize(kind);
  auto intType = builder.getIntegerType(bits);
  auto cast = builder.createConvert(loc, intType, code);
  auto undef = builder.create<fir::UndefOp>(loc, charType);
  auto zero = builder.getIntegerAttr(builder.getIndexType(), 0);
  return builder.create<fir::InsertValueOp>(loc, charType, undef, cast,
                                            builder.getArrayAttr(zero));
}

mlir::Value fir::factory::CharacterExprHelper::extractCodeFromSingleton(
    mlir::Value singleton) {
  auto type = getCharacterType(singleton);
  assert(type.getLen() == 1);
  auto bits = builder.getKindMap().getCharacterBitsize(type.getFKind());
  auto intType = builder.getIntegerType(bits);
  auto zero = builder.getIntegerAttr(builder.getIndexType(), 0);
  return builder.create<fir::ExtractValueOp>(loc, intType, singleton,
                                             builder.getArrayAttr(zero));
}

mlir::Value
fir::factory::CharacterExprHelper::readLengthFromBox(mlir::Value box) {
  auto lenTy = builder.getCharacterLengthType();
  auto size = builder.create<fir::BoxEleSizeOp>(loc, lenTy, box);
  auto charTy = recoverCharacterType(box.getType());
  auto bits = builder.getKindMap().getCharacterBitsize(charTy.getFKind());
  auto width = bits / 8;
  if (width > 1) {
    auto widthVal = builder.createIntegerConstant(loc, lenTy, width);
    return builder.create<mlir::arith::DivSIOp>(loc, size, widthVal);
  }
  return size;
}

mlir::Value fir::factory::CharacterExprHelper::getLength(mlir::Value memref) {
  auto memrefType = memref.getType();
  auto charType = recoverCharacterType(memrefType);
  assert(charType && "must be a character type");
  if (charType.hasConstantLen())
    return builder.createIntegerConstant(loc, builder.getCharacterLengthType(),
                                         charType.getLen());
  if (memrefType.isa<fir::BoxType>())
    return readLengthFromBox(memref);
  if (memrefType.isa<fir::BoxCharType>())
    return createUnboxChar(memref).second;

  // Length cannot be deduced from memref.
  return {};
}

std::pair<mlir::Value, mlir::Value>
fir::factory::extractCharacterProcedureTuple(fir::FirOpBuilder &builder,
                                             mlir::Location loc,
                                             mlir::Value tuple) {
  mlir::TupleType tupleType = tuple.getType().cast<mlir::TupleType>();
  mlir::Value addr = builder.create<fir::ExtractValueOp>(
      loc, tupleType.getType(0), tuple,
      builder.getArrayAttr(
          {builder.getIntegerAttr(builder.getIndexType(), 0)}));
  mlir::Value proc = [&]() -> mlir::Value {
    if (auto addrTy = addr.getType().dyn_cast<fir::BoxProcType>())
      return builder.create<fir::BoxAddrOp>(loc, addrTy.getEleTy(), addr);
    return addr;
  }();
  mlir::Value len = builder.create<fir::ExtractValueOp>(
      loc, tupleType.getType(1), tuple,
      builder.getArrayAttr(
          {builder.getIntegerAttr(builder.getIndexType(), 1)}));
  return {proc, len};
}

mlir::Value fir::factory::createCharacterProcedureTuple(
    fir::FirOpBuilder &builder, mlir::Location loc, mlir::Type argTy,
    mlir::Value addr, mlir::Value len) {
  mlir::TupleType tupleType = argTy.cast<mlir::TupleType>();
  addr = builder.createConvert(loc, tupleType.getType(0), addr);
  if (len)
    len = builder.createConvert(loc, tupleType.getType(1), len);
  else
    len = builder.create<fir::UndefOp>(loc, tupleType.getType(1));
  mlir::Value tuple = builder.create<fir::UndefOp>(loc, tupleType);
  tuple = builder.create<fir::InsertValueOp>(
      loc, tupleType, tuple, addr,
      builder.getArrayAttr(
          {builder.getIntegerAttr(builder.getIndexType(), 0)}));
  tuple = builder.create<fir::InsertValueOp>(
      loc, tupleType, tuple, len,
      builder.getArrayAttr(
          {builder.getIntegerAttr(builder.getIndexType(), 1)}));
  return tuple;
}

mlir::Type
fir::factory::getCharacterProcedureTupleType(mlir::Type funcPointerType) {
  mlir::MLIRContext *context = funcPointerType.getContext();
  mlir::Type lenType = mlir::IntegerType::get(context, 64);
  return mlir::TupleType::get(context, {funcPointerType, lenType});
}
