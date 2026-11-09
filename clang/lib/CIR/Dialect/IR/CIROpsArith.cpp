//===- CIROpsArith.cpp - CIR ops in partition `Arith` ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the CIR ops in partition `Arith`: the hand-written
// parts first, then the TableGen-generated definitions.
//
//===----------------------------------------------------------------------===//

#include "clang/CIR/Dialect/IR/CIROpsArith.h"

#include "CIROpDefinitions.h"

#include "mlir/IR/DialectImplementation.h"

using namespace mlir;
using namespace cir;

//===----------------------------------------------------------------------===//
// AddOp / SubOp
//===----------------------------------------------------------------------===//

// The integer-only type constraint on these ops makes the nsw/nuw/sat flag
// type checks unnecessary. Only the mutual-exclusivity between nsw/nuw and
// sat needs to be verified.

LogicalResult cir::AddOp::verify() {
  if (getSaturated() && (getNoSignedWrap() || getNoUnsignedWrap()))
    return emitOpError()
           << "the nsw/nuw flags and the saturated flag are mutually exclusive";
  return mlir::success();
}

LogicalResult cir::SubOp::verify() {
  if (getSaturated() && (getNoSignedWrap() || getNoUnsignedWrap()))
    return emitOpError()
           << "the nsw/nuw flags and the saturated flag are mutually exclusive";
  return mlir::success();
}

//===----------------------------------------------------------------------===//
// ShiftOp
//===----------------------------------------------------------------------===//
LogicalResult cir::ShiftOp::verify() {
  mlir::Operation *op = getOperation();
  auto op0VecTy = mlir::dyn_cast<cir::VectorType>(op->getOperand(0).getType());
  auto op1VecTy = mlir::dyn_cast<cir::VectorType>(op->getOperand(1).getType());
  if (!op0VecTy ^ !op1VecTy)
    return emitOpError() << "input types cannot be one vector and one scalar";

  if (op0VecTy) {
    if (op0VecTy.getSize() != op1VecTy.getSize())
      return emitOpError() << "input vector types must have the same size";

    auto opResultTy = mlir::dyn_cast<cir::VectorType>(getType());
    if (!opResultTy)
      return emitOpError() << "the type of the result must be a vector "
                           << "if it is vector shift";

    auto op0VecEleTy = mlir::cast<cir::IntType>(op0VecTy.getElementType());
    auto op1VecEleTy = mlir::cast<cir::IntType>(op1VecTy.getElementType());
    if (op0VecEleTy.getWidth() != op1VecEleTy.getWidth())
      return emitOpError()
             << "vector operands do not have the same elements sizes";

    auto resVecEleTy = mlir::cast<cir::IntType>(opResultTy.getElementType());
    if (op0VecEleTy.getWidth() != resVecEleTy.getWidth())
      return emitOpError() << "vector operands and result type do not have the "
                              "same elements sizes";
  }

  return mlir::success();
}

//===----------------------------------------------------------------------===//
// IncOp
//===----------------------------------------------------------------------===//

OpFoldResult cir::IncOp::fold(FoldAdaptor adaptor) {
  if (mlir::isa_and_present<cir::PoisonAttr>(adaptor.getInput()))
    return adaptor.getInput();
  return {};
}

//===----------------------------------------------------------------------===//
// DecOp
//===----------------------------------------------------------------------===//

OpFoldResult cir::DecOp::fold(FoldAdaptor adaptor) {
  if (mlir::isa_and_present<cir::PoisonAttr>(adaptor.getInput()))
    return adaptor.getInput();
  return {};
}

//===----------------------------------------------------------------------===//
// MinusOp
//===----------------------------------------------------------------------===//

OpFoldResult cir::MinusOp::fold(FoldAdaptor adaptor) {
  if (mlir::isa_and_present<cir::PoisonAttr>(adaptor.getInput()))
    return adaptor.getInput();

  // Fold with constant inputs. Floating-point negation is handled by
  // cir::FNegOp.
  if (auto intAttr =
          mlir::dyn_cast_if_present<cir::IntAttr>(adaptor.getInput())) {
    APInt val = intAttr.getValue();
    val.negate();
    return cir::IntAttr::get(getType(), val);
  }

  return {};
}

//===----------------------------------------------------------------------===//
// FNegOp
//===----------------------------------------------------------------------===//

OpFoldResult cir::FNegOp::fold(FoldAdaptor adaptor) {
  if (mlir::isa_and_present<cir::PoisonAttr>(adaptor.getInput()))
    return adaptor.getInput();

  // Fold with constant inputs.
  if (auto fpAttr =
          mlir::dyn_cast_if_present<cir::FPAttr>(adaptor.getInput())) {
    APFloat val = fpAttr.getValue();
    val.changeSign();
    return cir::FPAttr::get(getType(), val);
  }

  return {};
}

//===----------------------------------------------------------------------===//
// NotOp
//===----------------------------------------------------------------------===//

OpFoldResult cir::NotOp::fold(FoldAdaptor adaptor) {
  if (mlir::isa_and_present<cir::PoisonAttr>(adaptor.getInput()))
    return adaptor.getInput();

  // not(not(x)) -> x is handled by the Involution trait.

  // Fold with constant inputs.
  if (mlir::Attribute attr = adaptor.getInput()) {
    if (auto intAttr = mlir::dyn_cast<cir::IntAttr>(attr)) {
      APInt val = intAttr.getValue();
      val.flipAllBits();
      return cir::IntAttr::get(getType(), val);
    }
    if (auto boolAttr = mlir::dyn_cast<cir::BoolAttr>(attr))
      return cir::BoolAttr::get(getContext(), !boolAttr.getValue());
  }

  return {};
}

//===----------------------------------------------------------------------===//
// CmpOp
//===----------------------------------------------------------------------===//

LogicalResult cir::CmpOp::verify() {
  if (getFenvAttr() && !cir::isAnyFloatingPointType(getLhs().getType()))
    return emitOpError()
           << "'fenv' is only valid for floating-point comparisons";
  return success();
}

//===----------------------------------------------------------------------===//
// Bit manipulation operations
//===----------------------------------------------------------------------===//

static OpFoldResult
foldUnaryBitOp(mlir::Attribute inputAttr,
               llvm::function_ref<llvm::APInt(const llvm::APInt &)> func,
               bool poisonZero = false) {
  if (mlir::isa_and_present<cir::PoisonAttr>(inputAttr)) {
    // Propagate poison value
    return inputAttr;
  }

  auto input = mlir::dyn_cast_if_present<IntAttr>(inputAttr);
  if (!input)
    return nullptr;

  llvm::APInt inputValue = input.getValue();
  if (poisonZero && inputValue.isZero())
    return cir::PoisonAttr::get(input.getType());

  llvm::APInt resultValue = func(inputValue);
  return IntAttr::get(input.getType(), resultValue);
}

OpFoldResult BitClrsbOp::fold(FoldAdaptor adaptor) {
  return foldUnaryBitOp(adaptor.getInput(), [](const llvm::APInt &inputValue) {
    unsigned resultValue =
        inputValue.getBitWidth() - inputValue.getSignificantBits();
    return llvm::APInt(inputValue.getBitWidth(), resultValue);
  });
}

OpFoldResult BitClzOp::fold(FoldAdaptor adaptor) {
  return foldUnaryBitOp(
      adaptor.getInput(),
      [](const llvm::APInt &inputValue) {
        unsigned resultValue = inputValue.countLeadingZeros();
        return llvm::APInt(inputValue.getBitWidth(), resultValue);
      },
      getPoisonZero());
}

OpFoldResult BitCtzOp::fold(FoldAdaptor adaptor) {
  return foldUnaryBitOp(
      adaptor.getInput(),
      [](const llvm::APInt &inputValue) {
        return llvm::APInt(inputValue.getBitWidth(),
                           inputValue.countTrailingZeros());
      },
      getPoisonZero());
}

OpFoldResult BitFfsOp::fold(FoldAdaptor adaptor) {
  return foldUnaryBitOp(adaptor.getInput(), [](const llvm::APInt &inputValue) {
    unsigned trailingZeros = inputValue.countTrailingZeros();
    unsigned result =
        trailingZeros == inputValue.getBitWidth() ? 0 : trailingZeros + 1;
    return llvm::APInt(inputValue.getBitWidth(), result);
  });
}

OpFoldResult BitParityOp::fold(FoldAdaptor adaptor) {
  return foldUnaryBitOp(adaptor.getInput(), [](const llvm::APInt &inputValue) {
    return llvm::APInt(inputValue.getBitWidth(), inputValue.popcount() % 2);
  });
}

OpFoldResult BitPopcountOp::fold(FoldAdaptor adaptor) {
  return foldUnaryBitOp(adaptor.getInput(), [](const llvm::APInt &inputValue) {
    return llvm::APInt(inputValue.getBitWidth(), inputValue.popcount());
  });
}

OpFoldResult BitReverseOp::fold(FoldAdaptor adaptor) {
  return foldUnaryBitOp(adaptor.getInput(), [](const llvm::APInt &inputValue) {
    return inputValue.reverseBits();
  });
}

OpFoldResult ByteSwapOp::fold(FoldAdaptor adaptor) {
  return foldUnaryBitOp(adaptor.getInput(), [](const llvm::APInt &inputValue) {
    return inputValue.byteSwap();
  });
}

OpFoldResult RotateOp::fold(FoldAdaptor adaptor) {
  if (mlir::isa_and_present<cir::PoisonAttr>(adaptor.getInput()) ||
      mlir::isa_and_present<cir::PoisonAttr>(adaptor.getAmount())) {
    // Propagate poison values
    return cir::PoisonAttr::get(getType());
  }

  auto input = mlir::dyn_cast_if_present<IntAttr>(adaptor.getInput());
  auto amount = mlir::dyn_cast_if_present<IntAttr>(adaptor.getAmount());
  if (!input && !amount)
    return nullptr;

  // We could fold cir.rotate even if one of its two operands is not a constant:
  //   - `cir.rotate left/right %0, 0` could be folded into just %0 even if %0
  //     is not a constant.
  //   - `cir.rotate left/right 0/0b111...111, %0` could be folded into 0 or
  //     0b111...111 even if %0 is not a constant.

  llvm::APInt inputValue;
  if (input) {
    inputValue = input.getValue();
    if (inputValue.isZero() || inputValue.isAllOnes()) {
      // An input value of all 0s or all 1s will not change after rotation
      return input;
    }
  }

  uint64_t amountValue;
  if (amount) {
    amountValue = amount.getValue().urem(getInput().getType().getWidth());
    if (amountValue == 0) {
      // A shift amount of 0 will not change the input value
      return getInput();
    }
  }

  if (!input || !amount)
    return nullptr;

  assert(inputValue.getBitWidth() == getInput().getType().getWidth() &&
         "input value must have the same bit width as the input type");

  llvm::APInt resultValue;
  if (isRotateLeft())
    resultValue = inputValue.rotl(amountValue);
  else
    resultValue = inputValue.rotr(amountValue);

  return IntAttr::get(input.getContext(), input.getType(), resultValue);
}

//===----------------------------------------------------------------------===//
// TableGen-generated op definitions
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "clang/CIR/Dialect/IR/CIROpsArith.cpp.inc"
