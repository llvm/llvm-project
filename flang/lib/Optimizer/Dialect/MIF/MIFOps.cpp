//===-- MIFOps.cpp - MIF dialect ops implementation -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang/Optimizer/Dialect/MIF/MIFOps.h"
#include "flang/Optimizer/Builder/Todo.h"
#include "flang/Optimizer/Dialect/FIRAttr.h"
#include "flang/Optimizer/Dialect/FIRType.h"
#include "flang/Optimizer/Dialect/MIF/MIFDialect.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/ADT/SmallVector.h"

#define GET_OP_CLASSES
#include "flang/Optimizer/Dialect/MIF/MIFOps.cpp.inc"

//===----------------------------------------------------------------------===//
// NumImagesOp
//===----------------------------------------------------------------------===//

void mif::NumImagesOp::build(mlir::OpBuilder &builder,
                             mlir::OperationState &result,
                             mlir::Value teamArg) {
  bool isTeamNumber =
      teamArg && fir::unwrapPassByRefType(teamArg.getType()).isInteger();
  if (isTeamNumber)
    build(builder, result, teamArg, /*team*/ mlir::Value{});
  else
    build(builder, result, /*team_number*/ mlir::Value{}, teamArg);
}

llvm::LogicalResult mif::NumImagesOp::verify() {
  if (getTeam() && getTeamNumber())
    return emitOpError(
        "team and team_number must not be provided at the same time");
  return mlir::success();
}

//===----------------------------------------------------------------------===//
// ThisImageOp
//===----------------------------------------------------------------------===//

void mif::ThisImageOp::build(mlir::OpBuilder &builder,
                             mlir::OperationState &result, mlir::Value coarray,
                             mlir::Value team) {
  build(builder, result, coarray, /*dim*/ mlir::Value{}, team);
}

void mif::ThisImageOp::build(mlir::OpBuilder &builder,
                             mlir::OperationState &result, mlir::Value team) {
  build(builder, result, /*coarray*/ mlir::Value{}, /*dim*/ mlir::Value{},
        team);
}

llvm::LogicalResult mif::ThisImageOp::verify() {
  if (getDim() && !getCoarray())
    return emitOpError(
        "`dim` must be provied at the same time as the `coarray` argument.");
  return mlir::success();
}

//===----------------------------------------------------------------------===//
// SyncImagesOp
//===----------------------------------------------------------------------===//

llvm::LogicalResult mif::SyncImagesOp::verify() {
  if (getImageSet()) {
    mlir::Type t = getImageSet().getType();
    fir::BoxType boxTy = mlir::dyn_cast<fir::BoxType>(t);
    if (auto seqTy = mlir::dyn_cast<fir::SequenceType>(
            boxTy.getElementOrSequenceType())) {
      if (seqTy.getDimension() != 0 && seqTy.getDimension() != 1)
        return emitOpError(
            "`image_set` must be a boxed integer expression of rank 1.");
      if (!fir::isa_integer(seqTy.getElementType()))
        return emitOpError("`image_set` must be a boxed array of integer.");
    } else if (!fir::isa_integer(boxTy.getElementType()))
      return emitOpError(
          "`image_set` must be a boxed scalar integer expression.");
  }
  return mlir::success();
}

//===----------------------------------------------------------------------===//
// CoBroadcastOp
//===----------------------------------------------------------------------===//

llvm::LogicalResult mif::CoBroadcastOp::verify() {
  fir::BoxType boxTy = mlir::dyn_cast<fir::BoxType>(getA().getType());

  if (fir::isPolymorphicType(boxTy))
    return emitOpError("`A` cannot be polymorphic.");
  else if (auto recTy =
               mlir::dyn_cast<fir::RecordType>(boxTy.getElementType())) {
    for (auto component : recTy.getTypeList()) {
      if (fir::isPolymorphicType(component.second))
        TODO(getLoc(), "`A` with polymorphic subobject component.");
    }
  }
  return mlir::success();
}

//===----------------------------------------------------------------------===//
// CoMaxOp
//===----------------------------------------------------------------------===//

llvm::LogicalResult mif::CoMaxOp::verify() {
  fir::BoxType boxTy = mlir::dyn_cast<fir::BoxType>(getA().getType());
  mlir::Type elemTy = boxTy.getElementOrSequenceType();
  if (auto seqTy = mlir::dyn_cast<fir::SequenceType>(elemTy))
    elemTy = seqTy.getElementType();

  if (!fir::isa_real(elemTy) && !fir::isa_integer(elemTy) &&
      !fir::isa_char(elemTy))
    return emitOpError("`A` shall be of type integer, real or character.");
  return mlir::success();
}

//===----------------------------------------------------------------------===//
// CoMinOp
//===----------------------------------------------------------------------===//

llvm::LogicalResult mif::CoMinOp::verify() {
  fir::BoxType boxTy = mlir::dyn_cast<fir::BoxType>(getA().getType());
  mlir::Type elemTy = boxTy.getElementOrSequenceType();
  if (auto seqTy = mlir::dyn_cast<fir::SequenceType>(elemTy))
    elemTy = seqTy.getElementType();

  if (!fir::isa_real(elemTy) && !fir::isa_integer(elemTy) &&
      !fir::isa_char(elemTy))
    return emitOpError("`A` shall be of type integer, real or character.");
  return mlir::success();
}

//===----------------------------------------------------------------------===//
// CoSumOp
//===----------------------------------------------------------------------===//

llvm::LogicalResult mif::CoSumOp::verify() {
  fir::BoxType boxTy = mlir::dyn_cast<fir::BoxType>(getA().getType());
  mlir::Type elemTy = boxTy.getElementOrSequenceType();
  if (auto seqTy = mlir::dyn_cast<fir::SequenceType>(elemTy))
    elemTy = seqTy.getElementType();

  if (!fir::isa_real(elemTy) && !fir::isa_integer(elemTy) &&
      !fir::isa_complex(elemTy))
    return emitOpError("`A` shall be of numeric type.");
  return mlir::success();
}

//===----------------------------------------------------------------------===//
// ChangeTeamOp
//===----------------------------------------------------------------------===//

void mif::ChangeTeamOp::build(mlir::OpBuilder &builder,
                              mlir::OperationState &result, mlir::Value team,
                              bool ensureTerminator,
                              llvm::ArrayRef<mlir::NamedAttribute> attributes) {
  build(builder, result, team, /*stat*/ mlir::Value{}, /*errmsg*/ mlir::Value{},
        ensureTerminator, attributes);
}

void mif::ChangeTeamOp::build(mlir::OpBuilder &builder,
                              mlir::OperationState &result, mlir::Value team,
                              mlir::Value stat, mlir::Value errmsg,
                              bool ensureTerminator,
                              llvm::ArrayRef<mlir::NamedAttribute> attributes) {
  std::int32_t argStat = 0, argErrmsg = 0;
  result.addOperands(team);
  if (stat) {
    result.addOperands(stat);
    argStat++;
  }
  if (errmsg) {
    result.addOperands(errmsg);
    argErrmsg++;
  }

  mlir::Region *bodyRegion = result.addRegion();
  bodyRegion->push_back(new mlir::Block{});
  if (ensureTerminator)
    ChangeTeamOp::ensureTerminator(*bodyRegion, builder, result.location);

  result.addAttribute(getOperandSegmentSizeAttr(),
                      builder.getDenseI32ArrayAttr({1, argStat, argErrmsg}));
  result.addAttributes(attributes);
}

mlir::ParseResult mif::ChangeTeamOp::parse(mlir::OpAsmParser &parser,
                                           mlir::OperationState &result) {
  auto &builder = parser.getBuilder();
  llvm::SmallVector<mlir::OpAsmParser::UnresolvedOperand> opers;
  llvm::SmallVector<mlir::Type> types;
  int32_t statArg = 0, errmsgArg = 0;
  if (parser.parseOperand(opers.emplace_back()))
    return mlir::failure();

  if (mlir::succeeded(parser.parseOptionalKeyword("stat"))) {
    if (*parser.parseOptionalOperand(opers.emplace_back()))
      return mlir::failure();
    statArg++;
  }
  if (mlir::succeeded(parser.parseOptionalKeyword("errmsg"))) {
    if (*parser.parseOptionalOperand(opers.emplace_back()))
      return mlir::failure();
    errmsgArg++;
  }

  // Set the operandSegmentSizes attribute
  result.addAttribute(getOperandSegmentSizeAttr(),
                      builder.getDenseI32ArrayAttr({1, statArg, errmsgArg}));

  if (parser.parseColon())
    return mlir::failure();

  if (parser.parseLParen())
    return mlir::failure();
  if (parser.parseTypeList(types))
    return mlir::failure();
  if (parser.parseRParen())
    return mlir::failure();

  if (opers.size() != types.size())
    return mlir::failure();

  if (parser.resolveOperands(opers, types, parser.getCurrentLocation(),
                             result.operands))
    return mlir::failure();

  auto *body = result.addRegion();
  if (parser.parseRegion(*body))
    return mlir::failure();

  ChangeTeamOp::ensureTerminator(*body, builder, result.location);

  if (parser.parseOptionalAttrDictWithKeyword(result.attributes))
    return mlir::failure();

  return mlir::success();
}

void mif::ChangeTeamOp::print(mlir::OpAsmPrinter &p) {
  p << ' ' << getTeam();
  if (getStat())
    p << " stat " << getStat();
  if (getErrmsg())
    p << " errmsg " << getErrmsg();
  p << " : (";
  llvm::interleaveComma(getOperands(), p,
                        [&](mlir::Value v) { p << v.getType(); });
  p << ") ";
  p.printRegion(getRegion(), /*printEntryBlockArgs=*/true,
                /*printBlockTerminators=*/true);
  p.printOptionalAttrDict((*this)->getAttrs(),
                          {ChangeTeamOp::getOperandSegmentSizeAttr()});
}
