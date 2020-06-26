//===-- CharacterExpr.cpp -------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang/Lower/CharacterExpr.h"
#include "flang/Lower/ConvertType.h"
#include "flang/Lower/DoLoopHelper.h"
#include "flang/Lower/IntrinsicCall.h"

//===----------------------------------------------------------------------===//
// CharacterExprHelper implementation
//===----------------------------------------------------------------------===//

/// Get fir.char<kind> type with the same kind as inside str.
static fir::CharacterType getCharacterType(mlir::Type type) {
  if (auto boxType = type.dyn_cast<fir::BoxCharType>())
    return boxType.getEleTy();
  if (auto refType = type.dyn_cast<fir::ReferenceType>())
    type = refType.getEleTy();
  if (auto seqType = type.dyn_cast<fir::SequenceType>())
    type = seqType.getEleTy();
  if (auto charType = type.dyn_cast<fir::CharacterType>())
    return charType;
  llvm_unreachable("Invalid character value type");
}

static fir::CharacterType getCharacterType(const fir::CharBoxValue &box) {
  return getCharacterType(box.getBuffer().getType());
}

static bool needToMaterialize(const fir::CharBoxValue &box) {
  return box.getBuffer().getType().isa<fir::SequenceType>() ||
         box.getBuffer().getType().isa<fir::CharacterType>();
}

static std::optional<fir::SequenceType::Extent>
getCompileTimeLength(const fir::CharBoxValue &box) {
  // FIXME: should this just return box.getLen() ??
  auto type = box.getBuffer().getType();
  if (type.isa<fir::CharacterType>())
    return 1;
  if (auto refType = type.dyn_cast<fir::ReferenceType>())
    type = refType.getEleTy();
  if (auto seqType = type.dyn_cast<fir::SequenceType>()) {
    auto shape = seqType.getShape();
    assert(shape.size() == 1 && "only scalar character supported");
    if (shape[0] != fir::SequenceType::getUnknownExtent())
      return shape[0];
  }
  return {};
}

fir::CharBoxValue Fortran::lower::CharacterExprHelper::materializeValue(
    const fir::CharBoxValue &str) {
  if (!needToMaterialize(str))
    return str;
  auto variable = builder.create<fir::AllocaOp>(loc, str.getBuffer().getType());
  builder.create<fir::StoreOp>(loc, str.getBuffer(), variable);
  return {variable, str.getLen()};
}

fir::CharBoxValue
Fortran::lower::CharacterExprHelper::toDataLengthPair(mlir::Value character) {
  auto lenType = getLengthType();
  auto type = character.getType();
  if (auto boxCharType = type.dyn_cast<fir::BoxCharType>()) {
    auto refType = builder.getRefType(boxCharType.getEleTy());
    auto unboxed =
        builder.create<fir::UnboxCharOp>(loc, refType, lenType, character);
    return {unboxed.getResult(0), unboxed.getResult(1)};
  }
  if (auto seqType = type.dyn_cast<fir::CharacterType>()) {
    // Materialize length for usage into character manipulations.
    auto len = builder.createIntegerConstant(loc, lenType, 1);
    return {character, len};
  }
  if (auto refType = type.dyn_cast<fir::ReferenceType>())
    type = refType.getEleTy();
  if (auto seqType = type.dyn_cast<fir::SequenceType>()) {
    assert(seqType.hasConstantShape() &&
           "ssa array value must have constant length");
    auto shape = seqType.getShape();
    assert(shape.size() == 1 && "only scalar character supported");
    // Materialize length for usage into character manipulations.
    auto len = builder.createIntegerConstant(loc, lenType, shape[0]);
    // FIXME: this seems to work for tests, but don't think it is correct
    if (auto load = dyn_cast<fir::LoadOp>(character.getDefiningOp()))
      return {load.memref(), len};
    return {character, len};
  }
  if (auto charTy = type.dyn_cast<fir::CharacterType>()) {
    auto len = builder.createIntegerConstant(loc, lenType, 1);
    return {character, len};
  }
  llvm::report_fatal_error("unexpected character type");
}

/// Get fir.ref<fir.char<kind>> type.
mlir::Type Fortran::lower::CharacterExprHelper::getReferenceType(
    const fir::CharBoxValue &box) const {
  return builder.getRefType(getCharacterType(box));
}

mlir::Value
Fortran::lower::CharacterExprHelper::createEmbox(const fir::CharBoxValue &box) {
  // BoxChar require a reference.
  auto str = box;
  if (needToMaterialize(box))
    str = materializeValue(box);
  auto kind = getCharacterType(str).getFKind();
  auto boxCharType = fir::BoxCharType::get(builder.getContext(), kind);
  auto refType = getReferenceType(str);
  // So far, fir.emboxChar fails lowering to llvm when it is given
  // fir.data<fir.array<len x fir.char<kind>>> types, so convert to
  // fir.data<fir.char<kind>> if needed.
  auto buff = str.getBuffer();
  if (refType != str.getBuffer().getType())
    buff = builder.createConvert(loc, refType, buff);
  // Convert in case the provided length is not of the integer type that must
  // be used in boxchar.
  auto lenType = getLengthType();
  auto len = str.getLen();
  if (str.getLen().getType() != lenType)
    len = builder.createConvert(loc, lenType, len);
  return builder.create<fir::EmboxCharOp>(loc, boxCharType, buff, len);
}

mlir::Value Fortran::lower::CharacterExprHelper::createLoadCharAt(
    const fir::CharBoxValue &str, mlir::Value index) {
  // In case this is addressing a length one character scalar simply return
  // the single character.
  if (str.getBuffer().getType().isa<fir::CharacterType>())
    return str.getBuffer();
  auto addr = builder.create<fir::CoordinateOp>(loc, getReferenceType(str),
                                                str.getBuffer(), index);
  return builder.create<fir::LoadOp>(loc, addr);
}

void Fortran::lower::CharacterExprHelper::createStoreCharAt(
    const fir::CharBoxValue &str, mlir::Value index, mlir::Value c) {
  assert(!needToMaterialize(str) && "not in memory");
  auto addr = builder.create<fir::CoordinateOp>(loc, getReferenceType(str),
                                                str.getBuffer(), index);
  builder.create<fir::StoreOp>(loc, c, addr);
}

void Fortran::lower::CharacterExprHelper::createCopy(
    const fir::CharBoxValue &dest, const fir::CharBoxValue &src,
    mlir::Value count) {
  Fortran::lower::DoLoopHelper{builder, loc}.createLoop(
      count, [&](Fortran::lower::FirOpBuilder &, mlir::Value index) {
        auto charVal = createLoadCharAt(src, index);
        createStoreCharAt(dest, index, charVal);
      });
}

void Fortran::lower::CharacterExprHelper::createPadding(
    const fir::CharBoxValue &str, mlir::Value lower, mlir::Value upper) {
  auto blank = createBlankConstant(getCharacterType(str));
  // Always create the loop, if upper < lower, no iteration will be
  // executed.
  Fortran::lower::DoLoopHelper{builder, loc}.createLoop(
      lower, upper, [&](Fortran::lower::FirOpBuilder &, mlir::Value index) {
        createStoreCharAt(str, index, blank);
      });
}

fir::CharBoxValue
Fortran::lower::CharacterExprHelper::createTemp(mlir::Type type,
                                                mlir::Value len) {
  assert(type.isa<fir::CharacterType>() && "expected fir character type");
  llvm::SmallVector<mlir::Value, 3> sizes{len};
  auto ref = builder.allocateLocal(loc, type, llvm::StringRef{}, sizes);
  return {ref, len};
}

// Simple length one character assignment without loops.
void Fortran::lower::CharacterExprHelper::createLengthOneAssign(
    const fir::CharBoxValue &lhs, const fir::CharBoxValue &rhs) {
  auto addr = lhs.getBuffer();
  auto refType = getReferenceType(lhs);
  addr = builder.createConvert(loc, refType, addr);

  auto val = rhs.getBuffer();
  if (!needToMaterialize(rhs)) {
    mlir::Value rhsAddr = rhs.getBuffer();
    rhsAddr = builder.createConvert(loc, refType, rhsAddr);
    val = builder.create<fir::LoadOp>(loc, rhsAddr);
  }

  builder.create<fir::StoreOp>(loc, val, addr);
}

void Fortran::lower::CharacterExprHelper::createAssign(
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
  mlir::Value copyCount = lhs.getLen();
  if (!compileTimeSameLength)
    copyCount = Fortran::lower::IntrinsicCallOpsHelper{builder, loc}.genMin(
        {lhs.getLen(), rhs.getLen()});

  fir::CharBoxValue safeRhs = rhs;
  if (needToMaterialize(rhs)) {
    // TODO: revisit now that character constant handling changed.
    // Need to materialize the constant to get its elements.
    // (No equivalent of fir.coordinate_of for array value).
    safeRhs = materializeValue(rhs);
  } else {
    // If rhs is in memory, always assumes rhs might overlap with lhs
    // in a way that require a temp for the copy. That can be optimize later.
    // Only create a temp of copyCount size because we do not need more from
    // rhs.
    auto temp = createTemp(getCharacterType(rhs), copyCount);
    createCopy(temp, rhs, copyCount);
    safeRhs = temp;
  }

  // Actual copy
  createCopy(lhs, safeRhs, copyCount);

  // Pad if needed.
  if (!compileTimeSameLength) {
    auto one = builder.createIntegerConstant(loc, lhs.getLen().getType(), 1);
    auto maxPadding = builder.create<mlir::SubIOp>(loc, lhs.getLen(), one);
    createPadding(lhs, copyCount, maxPadding);
  }
}

fir::CharBoxValue Fortran::lower::CharacterExprHelper::createConcatenate(
    const fir::CharBoxValue &lhs, const fir::CharBoxValue &rhs) {
  mlir::Value len =
      builder.create<mlir::AddIOp>(loc, lhs.getLen(), rhs.getLen());
  auto temp = createTemp(getCharacterType(rhs), len);
  createCopy(temp, lhs, lhs.getLen());
  auto one = builder.createIntegerConstant(loc, len.getType(), 1);
  auto upperBound = builder.create<mlir::SubIOp>(loc, len, one);
  auto lhsLen =
      builder.createConvert(loc, builder.getIndexType(), lhs.getLen());
  Fortran::lower::DoLoopHelper{builder, loc}.createLoop(
      lhs.getLen(), upperBound, one,
      [&](Fortran::lower::FirOpBuilder &bldr, mlir::Value index) {
        auto rhsIndex = bldr.create<mlir::SubIOp>(loc, index, lhsLen);
        auto charVal = createLoadCharAt(rhs, rhsIndex);
        createStoreCharAt(temp, index, charVal);
      });
  return temp;
}

fir::CharBoxValue Fortran::lower::CharacterExprHelper::createSubstring(
    const fir::CharBoxValue &box, llvm::ArrayRef<mlir::Value> bounds) {
  // Constant need to be materialize in memory to use fir.coordinate_of.
  auto str = box;
  if (needToMaterialize(box))
    str = materializeValue(box);

  auto nbounds{bounds.size()};
  if (nbounds < 1 || nbounds > 2) {
    mlir::emitError(loc, "Incorrect number of bounds in substring");
    return {mlir::Value{}, mlir::Value{}};
  }
  mlir::SmallVector<mlir::Value, 2> castBounds;
  // Convert bounds to length type to do safe arithmetic on it.
  for (auto bound : bounds)
    castBounds.push_back(builder.createConvert(loc, getLengthType(), bound));
  auto lowerBound = castBounds[0];
  // FIR CoordinateOp is zero based but Fortran substring are one based.
  auto one = builder.createIntegerConstant(loc, lowerBound.getType(), 1);
  auto offset = builder.create<mlir::SubIOp>(loc, lowerBound, one).getResult();
  auto idxType = builder.getIndexType();
  if (offset.getType() != idxType)
    offset = builder.createConvert(loc, idxType, offset);
  auto substringRef = builder.create<fir::CoordinateOp>(
      loc, getReferenceType(str), str.getBuffer(), offset);

  // Compute the length.
  mlir::Value substringLen{};
  if (nbounds < 2) {
    substringLen =
        builder.create<mlir::SubIOp>(loc, str.getLen(), castBounds[0]);
  } else {
    substringLen =
        builder.create<mlir::SubIOp>(loc, castBounds[1], castBounds[0]);
  }
  substringLen = builder.create<mlir::AddIOp>(loc, substringLen, one);

  // Set length to zero if bounds were reversed (Fortran 2018 9.4.1)
  auto zero = builder.createIntegerConstant(loc, substringLen.getType(), 0);
  auto cdt = builder.create<mlir::CmpIOp>(loc, mlir::CmpIPredicate::slt,
                                          substringLen, zero);
  substringLen = builder.create<mlir::SelectOp>(loc, cdt, zero, substringLen);

  return {substringRef, substringLen};
}

mlir::Value Fortran::lower::CharacterExprHelper::createLenTrim(
    const fir::CharBoxValue &str) {
  // Note: Runtime for LEN_TRIM should also be available at some
  // point. For now use an inlined implementation.
  auto indexType = builder.getIndexType();
  auto len = builder.createConvert(loc, indexType, str.getLen());
  auto one = builder.createIntegerConstant(loc, indexType, 1);
  auto minusOne = builder.createIntegerConstant(loc, indexType, -1);
  auto zero = builder.createIntegerConstant(loc, indexType, 0);
  auto trueVal = builder.createIntegerConstant(loc, builder.getI1Type(), 1);
  auto blank = createBlankConstantCode(getCharacterType(str));
  mlir::Value lastChar = builder.create<mlir::SubIOp>(loc, len, one);

  auto iterWhile = builder.create<fir::IterWhileOp>(
      loc, lastChar, zero, minusOne, trueVal, lastChar);
  auto insPt = builder.saveInsertionPoint();
  builder.setInsertionPointToStart(iterWhile.getBody());
  auto index = iterWhile.getInductionVar();
  // Look for first non-blank from the right of the character.
  auto c = createLoadCharAt(str, index);
  c = builder.createConvert(loc, blank.getType(), c);
  auto isBlank =
      builder.create<mlir::CmpIOp>(loc, mlir::CmpIPredicate::eq, blank, c);
  llvm::SmallVector<mlir::Value, 2> results = {isBlank, index};
  builder.create<fir::ResultOp>(loc, results);
  builder.restoreInsertionPoint(insPt);
  // Compute length after iteration (zero if all blanks)
  mlir::Value newLen =
      builder.create<mlir::AddIOp>(loc, iterWhile.getResult(1), one);
  auto result =
      builder.create<SelectOp>(loc, iterWhile.getResult(0), zero, newLen);
  return builder.createConvert(loc, getLengthType(), result);
}

mlir::Value Fortran::lower::CharacterExprHelper::createTemp(mlir::Type type,
                                                            int len) {
  assert(type.isa<fir::CharacterType>() && "expected fir character type");
  assert(len >= 0 && "expected positive length");
  fir::SequenceType::Shape shape{len};
  auto seqType = fir::SequenceType::get(shape, type);
  return builder.create<fir::AllocaOp>(loc, seqType);
}

// Returns integer with code for blank. The integer has the same
// size as the character. Blank has ascii space code for all kinds.
mlir::Value Fortran::lower::CharacterExprHelper::createBlankConstantCode(
    fir::CharacterType type) {
  auto bits = builder.getKindMap().getCharacterBitsize(type.getFKind());
  auto intType = builder.getIntegerType(bits);
  return builder.createIntegerConstant(loc, intType, ' ');
}

mlir::Value Fortran::lower::CharacterExprHelper::createBlankConstant(
    fir::CharacterType type) {
  return builder.createConvert(loc, type, createBlankConstantCode(type));
}

void Fortran::lower::CharacterExprHelper::createCopy(mlir::Value dest,
                                                     mlir::Value src,
                                                     mlir::Value count) {
  createCopy(toDataLengthPair(dest), toDataLengthPair(src), count);
}

void Fortran::lower::CharacterExprHelper::createPadding(mlir::Value str,
                                                        mlir::Value lower,
                                                        mlir::Value upper) {
  createPadding(toDataLengthPair(str), lower, upper);
}

mlir::Value Fortran::lower::CharacterExprHelper::createSubstring(
    mlir::Value str, llvm::ArrayRef<mlir::Value> bounds) {
  return createEmbox(createSubstring(toDataLengthPair(str), bounds));
}

void Fortran::lower::CharacterExprHelper::createAssign(mlir::Value lhs,
                                                       mlir::Value rhs) {
  createAssign(toDataLengthPair(lhs), toDataLengthPair(rhs));
}

mlir::Value
Fortran::lower::CharacterExprHelper::createLenTrim(mlir::Value str) {
  return createLenTrim(toDataLengthPair(str));
}

void Fortran::lower::CharacterExprHelper::createAssign(mlir::Value lptr,
                                                       mlir::Value llen,
                                                       mlir::Value rptr,
                                                       mlir::Value rlen) {
  createAssign(fir::CharBoxValue{lptr, llen}, fir::CharBoxValue{rptr, rlen});
}

mlir::Value
Fortran::lower::CharacterExprHelper::createConcatenate(mlir::Value lhs,
                                                       mlir::Value rhs) {
  return createEmbox(
      createConcatenate(toDataLengthPair(lhs), toDataLengthPair(rhs)));
}

mlir::Value
Fortran::lower::CharacterExprHelper::createEmboxChar(mlir::Value addr,
                                                     mlir::Value len) {
  return createEmbox(fir::CharBoxValue{addr, len});
}

std::pair<mlir::Value, mlir::Value>
Fortran::lower::CharacterExprHelper::createUnboxChar(mlir::Value boxChar) {
  auto box = toDataLengthPair(boxChar);
  return {box.getBuffer(), box.getLen()};
}

mlir::Value
Fortran::lower::CharacterExprHelper::createCharacterTemp(mlir::Type type,
                                                         mlir::Value len) {
  return createEmbox(createTemp(type, len));
}

std::pair<mlir::Value, mlir::Value>
Fortran::lower::CharacterExprHelper::materializeCharacter(mlir::Value str) {
  auto box = toDataLengthPair(str);
  if (needToMaterialize(box))
    box = materializeValue(box);
  return {box.getBuffer(), box.getLen()};
}

bool Fortran::lower::CharacterExprHelper::isCharacterLiteral(mlir::Type type) {
  if (auto seqType = type.dyn_cast<fir::SequenceType>())
    return seqType.getEleTy().isa<fir::CharacterType>();
  return false;
}

bool Fortran::lower::CharacterExprHelper::isCharacter(mlir::Type type) {
  if (type.isa<fir::BoxCharType>())
    return true;
  if (auto refType = type.dyn_cast<fir::ReferenceType>())
    type = refType.getEleTy();
  if (auto seqType = type.dyn_cast<fir::SequenceType>()) {
    type = seqType.getEleTy();
  }
  return type.isa<fir::CharacterType>();
}

int Fortran::lower::CharacterExprHelper::getCharacterKind(mlir::Type type) {
  return getCharacterType(type).getFKind();
}
